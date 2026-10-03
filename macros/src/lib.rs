//! `#[function_tool]` attribute macro for openai-agents.

use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{
    parse_macro_input, punctuated::Punctuated, Attribute, Expr, FnArg, ItemFn, Lit, Meta, Pat,
    ReturnType, Token, Type,
};

/// Turn an async/sync function into an `openai_agents::FunctionTool` constructor.
///
/// The function's parameters are collected into a hidden arguments struct that derives
/// `serde::Deserialize` and `schemars::JsonSchema`, so nested types, `Vec<T>`, `Option<T>`
/// and user structs all produce a complete JSON Schema. The schema is then normalized with
/// `openai_agents::strict_schema::ensure_strict_json_schema`, matching the Python SDK's
/// `ensure_strict_json_schema`.
///
/// ```ignore
/// /// Add two numbers.
/// #[function_tool]
/// async fn add(a: i64, b: i64) -> i64 { a + b }
///
/// let tool = add(); // -> FunctionTool
/// ```
#[proc_macro_attribute]
pub fn function_tool(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr with Punctuated::<Meta, Token![,]>::parse_terminated);
    let mut name_override: Option<String> = None;
    let mut description_override: Option<String> = None;
    let mut needs_approval = false;
    for meta in args {
        match meta {
            Meta::NameValue(nv) if nv.path.is_ident("name") => {
                name_override = lit_str(&nv.value);
            }
            Meta::NameValue(nv) if nv.path.is_ident("description") => {
                description_override = lit_str(&nv.value);
            }
            Meta::NameValue(nv) if nv.path.is_ident("needs_approval") => {
                needs_approval = lit_bool(&nv.value).unwrap_or(false);
            }
            Meta::Path(p) if p.is_ident("needs_approval") => {
                needs_approval = true;
            }
            _ => {}
        }
    }

    let input_fn = parse_macro_input!(item as ItemFn);
    let vis = &input_fn.vis;
    let sig = &input_fn.sig;
    let is_async = sig.asyncness.is_some();
    let fn_ident = &sig.ident;
    let impl_ident = format_ident!("{fn_ident}_impl");
    // Python names the generated args model `{func_name}_args`; keep the same title.
    let args_ident = format_ident!("{fn_ident}_args");
    let tool_name = name_override.unwrap_or_else(|| fn_ident.to_string());

    // Fall back to the function's doc comment for the tool description (Python: docstring).
    let doc_attrs: Vec<&Attribute> = input_fn
        .attrs
        .iter()
        .filter(|a| a.path().is_ident("doc"))
        .collect();
    let doc_description = doc_attrs
        .first()
        .and_then(|a| match &a.meta {
            Meta::NameValue(nv) => lit_str(&nv.value),
            _ => None,
        })
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty());
    let tool_description = description_override.or(doc_description).unwrap_or_default();

    // Split parameters into an optional leading context parameter and the data parameters
    // that become the tool's JSON Schema properties.
    let mut takes_context = false;
    let mut fields = Vec::new();
    let mut field_names = Vec::new();

    for (index, arg) in sig.inputs.iter().enumerate() {
        let FnArg::Typed(pat_ty) = arg else {
            return syn::Error::new_spanned(arg, "function_tool does not support `self` methods")
                .to_compile_error()
                .into();
        };

        if is_context_type(&pat_ty.ty) {
            if index != 0 {
                return syn::Error::new_spanned(
                    &pat_ty.ty,
                    "RunContextWrapper/ToolContext param found at non-first position",
                )
                .to_compile_error()
                .into();
            }
            if matches!(*pat_ty.ty, Type::Reference(_)) {
                return syn::Error::new_spanned(
                    &pat_ty.ty,
                    "function_tool context param must be owned, not a reference; use `ToolContext`",
                )
                .to_compile_error()
                .into();
            }
            takes_context = true;
            continue;
        }

        let Pat::Ident(pat_ident) = &*pat_ty.pat else {
            return syn::Error::new_spanned(
                &pat_ty.pat,
                "function_tool only supports plain ident params",
            )
            .to_compile_error()
            .into();
        };

        if let Type::Reference(r) = &*pat_ty.ty {
            return syn::Error::new_spanned(
                r,
                format!(
                    "function_tool param `{}` is a reference type; use an owned type such as `String`",
                    pat_ident.ident
                ),
            )
            .to_compile_error()
            .into();
        }

        let pname = &pat_ident.ident;
        let pty = &*pat_ty.ty;
        let description = param_doc(&pat_ty.attrs);
        let description_attr = description.map(|d| quote! { #[schemars(description = #d)] });

        fields.push(quote! {
            #description_attr
            #pname: #pty
        });
        field_names.push(pname.clone());
    }

    let call = {
        let ctx_arg = if takes_context {
            quote! { __ctx, }
        } else {
            quote! {}
        };
        let body = if is_async {
            quote! { #impl_ident(#ctx_arg #(#field_names),*).await }
        } else {
            quote! { #impl_ident(#ctx_arg #(#field_names),*) }
        };
        body
    };

    let destructured = if field_names.is_empty() {
        quote! {}
    } else {
        quote! { let #args_ident { #(#field_names),* } = __parsed; }
    };

    let ctx_pat = if takes_context {
        quote! { __ctx }
    } else {
        quote! { _ctx }
    };

    let mut impl_fn = input_fn.clone();
    impl_fn.sig.ident = impl_ident.clone();
    impl_fn.vis = syn::Visibility::Inherited;
    // The doc comment belongs on the generated constructor, not on the hidden impl.
    impl_fn.attrs.retain(|a| !a.path().is_ident("doc"));

    let returns_unit = matches!(sig.output, ReturnType::Default);
    let result_to_value = if returns_unit {
        quote! { Ok(serde_json::Value::Null) }
    } else {
        quote! {
            Ok(serde_json::to_value(__out)
                .map_err(|e| openai_agents::AgentsError::tool_with_source("could not serialize tool output", e))?)
        }
    };

    let expanded = quote! {
        #impl_fn

        #(#doc_attrs)*
        #vis fn #fn_ident() -> openai_agents::FunctionTool {
            #[derive(openai_agents::serde::Deserialize, openai_agents::schemars::JsonSchema)]
            #[allow(non_camel_case_types, dead_code)]
            struct #args_ident {
                #(#fields),*
            }

            let __raw_schema = serde_json::to_value(openai_agents::schemars::schema_for!(#args_ident))
                .expect("function_tool: could not serialize the parameter JSON schema");
            let mut __schema =
                openai_agents::strict_schema::ensure_strict_json_schema(&__raw_schema)
                    .expect("function_tool: parameters cannot be converted to a strict JSON schema");
            // Python's pydantic schemas do not carry a `$schema` keyword; drop it so the
            // advertised tool schema matches the reference implementation byte for byte.
            if let Some(__root) = __schema.as_object_mut() {
                __root.remove("$schema");
            }

            openai_agents::FunctionTool::new(
                #tool_name,
                #tool_description,
                __schema,
                move |#ctx_pat, args| {
                    async move {
                        let __parsed: #args_ident = serde_json::from_str(&args).map_err(|e| {
                            openai_agents::AgentsError::tool_with_source(
                                format!("invalid arguments for tool `{}`", #tool_name),
                                e,
                            )
                        })?;
                        #destructured
                        let __out = #call;
                        #result_to_value
                    }
                },
            )
            .with_needs_approval(#needs_approval)
        }
    };

    TokenStream::from(expanded)
}

/// Whether a parameter type is a run/tool context that the SDK injects.
fn is_context_type(ty: &Type) -> bool {
    let inner = match ty {
        Type::Reference(r) => &*r.elem,
        other => other,
    };
    let Type::Path(p) = inner else {
        return false;
    };
    matches!(
        p.path.segments.last().map(|s| s.ident.to_string()).as_deref(),
        Some("ToolContext" | "RunContextWrapper")
    )
}

/// Doc comment attached to a parameter, if any.
fn param_doc(attrs: &[Attribute]) -> Option<String> {
    let text = attrs
        .iter()
        .filter(|a| a.path().is_ident("doc"))
        .filter_map(|a| match &a.meta {
            Meta::NameValue(nv) => lit_str(&nv.value),
            _ => None,
        })
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect::<Vec<_>>()
        .join(" ");
    if text.is_empty() {
        None
    } else {
        Some(text)
    }
}

fn lit_str(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Lit(el) => match &el.lit {
            Lit::Str(s) => Some(s.value()),
            _ => None,
        },
        _ => None,
    }
}

fn lit_bool(expr: &Expr) -> Option<bool> {
    match expr {
        Expr::Lit(el) => match &el.lit {
            Lit::Bool(b) => Some(b.value()),
            _ => None,
        },
        _ => None,
    }
}
