//! `#[function_tool]` attribute macro for openai-agents.

use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{
    parse_macro_input, punctuated::Punctuated, Expr, FnArg, ItemFn, Lit, Meta, Pat, ReturnType,
    Token, Type,
};

/// Turn an async/sync function into an `openai_agents::FunctionTool` constructor.
///
/// ```ignore
/// #[function_tool(description = "Add two numbers")]
/// async fn add(a: i64, b: i64) -> i64 { a + b }
///
/// let tool = add(); // -> FunctionTool
/// ```
#[proc_macro_attribute]
pub fn function_tool(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr with Punctuated::<Meta, Token![,]>::parse_terminated);
    let mut name_override: Option<String> = None;
    let mut description: Option<String> = None;
    for meta in args {
        match meta {
            Meta::NameValue(nv) if nv.path.is_ident("name") => {
                name_override = lit_str(&nv.value);
            }
            Meta::NameValue(nv) if nv.path.is_ident("description") => {
                description = lit_str(&nv.value);
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
    let tool_name = name_override.unwrap_or_else(|| fn_ident.to_string());
    let tool_description = description.unwrap_or_default();

    let mut param_names = Vec::new();
    let mut param_types = Vec::new();
    let mut schema_props = Vec::new();
    let mut required = Vec::new();

    for arg in &sig.inputs {
        let FnArg::Typed(pat_ty) = arg else {
            continue; // skip receiver
        };
        let Pat::Ident(pat_ident) = &*pat_ty.pat else {
            return syn::Error::new_spanned(&pat_ty.pat, "function_tool only supports plain ident params")
                .to_compile_error()
                .into();
        };
        let pname = &pat_ident.ident;
        let pty = &*pat_ty.ty;
        let (json_ty, optional) = map_json_type(pty);
        param_names.push(pname.clone());
        param_types.push(pty.clone());
        let pname_str = pname.to_string();
        schema_props.push(quote! {
            props.insert(#pname_str.to_string(), serde_json::json!({"type": #json_ty}));
        });
        if !optional {
            required.push(pname_str);
        }
    }

    let extractors: Vec<_> = param_names
        .iter()
        .zip(param_types.iter())
        .map(|(pname, pty)| {
            let pname_str = pname.to_string();
            let extract = extract_expr(pty, &pname_str);
            quote! {
                let #pname: #pty = #extract;
            }
        })
        .collect();

    let call = if is_async {
        quote! { #impl_ident(#(#param_names),*).await }
    } else {
        quote! { #impl_ident(#(#param_names),*) }
    };

    let mut impl_fn = input_fn.clone();
    impl_fn.sig.ident = impl_ident.clone();
    impl_fn.vis = syn::Visibility::Inherited;

    let returns_unit = matches!(sig.output, ReturnType::Default);
    let result_to_value = if returns_unit {
        quote! { Ok(serde_json::Value::Null) }
    } else {
        quote! {
            Ok(serde_json::to_value(__out)
                .map_err(|e| openai_agents::AgentsError::Tool(e.to_string()))?)
        }
    };

    let expanded = quote! {
        #impl_fn

        #vis fn #fn_ident() -> openai_agents::FunctionTool {
            let mut props = serde_json::Map::new();
            #(#schema_props)*
            let schema = serde_json::json!({
                "type": "object",
                "properties": props,
                "required": [#(#required),*],
                "additionalProperties": false
            });
            openai_agents::FunctionTool::new(
                #tool_name,
                #tool_description,
                schema,
                move |_ctx, args| {
                    async move {
                        let __v: serde_json::Value = serde_json::from_str(&args)
                            .map_err(|e| openai_agents::AgentsError::Tool(e.to_string()))?;
                        #(#extractors)*
                        let __out = #call;
                        #result_to_value
                    }
                },
            )
        }
    };

    TokenStream::from(expanded)
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

fn map_json_type(ty: &Type) -> (&'static str, bool) {
    if let Type::Path(p) = ty {
        if let Some(seg) = p.path.segments.last() {
            let name = seg.ident.to_string();
            if name == "Option" {
                if let syn::PathArguments::AngleBracketed(ab) = &seg.arguments {
                    if let Some(syn::GenericArgument::Type(inner)) = ab.args.first() {
                        let (t, _) = map_json_type(inner);
                        return (t, true);
                    }
                }
            }
            return (
                match name.as_str() {
                    "String" | "str" => "string",
                    "bool" => "boolean",
                    "i8" | "i16" | "i32" | "i64" | "u8" | "u16" | "u32" | "u64" | "isize"
                    | "usize" => "integer",
                    "f32" | "f64" => "number",
                    _ => "object",
                },
                false,
            );
        }
    }
    if let Type::Reference(r) = ty {
        return map_json_type(&r.elem);
    }
    ("object", false)
}

fn extract_expr(ty: &Type, key: &str) -> proc_macro2::TokenStream {
    if let Type::Path(p) = ty {
        if let Some(seg) = p.path.segments.last() {
            let name = seg.ident.to_string();
            if name == "Option" {
                return quote! {
                    __v.get(#key).cloned().and_then(|x| serde_json::from_value(x).ok())
                };
            }
            return match name.as_str() {
                "String" => quote! {
                    __v.get(#key)
                        .and_then(|x| x.as_str())
                        .ok_or_else(|| openai_agents::AgentsError::Tool(format!("missing string arg `{}`", #key)))?
                        .to_string()
                },
                "bool" => quote! {
                    __v.get(#key)
                        .and_then(|x| x.as_bool())
                        .ok_or_else(|| openai_agents::AgentsError::Tool(format!("missing bool arg `{}`", #key)))?
                },
                "i8" | "i16" | "i32" | "i64" | "isize" => quote! {
                    __v.get(#key)
                        .and_then(|x| x.as_i64())
                        .ok_or_else(|| openai_agents::AgentsError::Tool(format!("missing integer arg `{}`", #key)))? as #ty
                },
                "u8" | "u16" | "u32" | "u64" | "usize" => quote! {
                    __v.get(#key)
                        .and_then(|x| x.as_u64())
                        .ok_or_else(|| openai_agents::AgentsError::Tool(format!("missing integer arg `{}`", #key)))? as #ty
                },
                "f32" | "f64" => quote! {
                    __v.get(#key)
                        .and_then(|x| x.as_f64())
                        .ok_or_else(|| openai_agents::AgentsError::Tool(format!("missing number arg `{}`", #key)))? as #ty
                },
                _ => quote! {
                    serde_json::from_value(
                        __v.get(#key)
                            .cloned()
                            .ok_or_else(|| openai_agents::AgentsError::Tool(format!("missing arg `{}`", #key)))?
                    ).map_err(|e| openai_agents::AgentsError::Tool(e.to_string()))?
                },
            };
        }
    }
    quote! {
        serde_json::from_value(
            __v.get(#key)
                .cloned()
                .ok_or_else(|| openai_agents::AgentsError::Tool(format!("missing arg `{}`", #key)))?
        ).map_err(|e| openai_agents::AgentsError::Tool(e.to_string()))?
    }
}
