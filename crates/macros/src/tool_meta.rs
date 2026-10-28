//! `#[derive(ToolMeta)]`: a tool's name, description and policy, declared on its argument struct.

use proc_macro2::TokenStream;
use quote::quote;
use syn::{
    punctuated::Punctuated, DeriveInput, Expr, ExprPath, Lit, Meta, MetaNameValue, Token, Type,
};

use crate::{expect_seconds, expect_str, lit_bool};

const SUPPORTED_OPTIONS: &str = "`name`, `description`, `needs_approval`, `needs_approval_if`, \
                                 `timeout`, `on_failure`, `requires(..)`";

/// Options accepted by `#[tool(...)]`. Anything else is a compile error, for the same reason as in
/// `#[function_tool]`: an accepted-but-ignored option would run the tool without the limit or the
/// pause the author asked for.
#[derive(Default)]
pub(crate) struct MetaAttrs {
    pub(crate) name: Option<String>,
    pub(crate) description: Option<String>,
    pub(crate) needs_approval: bool,
    pub(crate) needs_approval_if: Option<ExprPath>,
    pub(crate) timeout_seconds: Option<f64>,
    pub(crate) on_failure: Option<syn::LitStr>,
    pub(crate) requires: Vec<Type>,
}

impl MetaAttrs {
    pub(crate) fn parse(args: Punctuated<Meta, Token![,]>) -> syn::Result<Self> {
        let mut out = Self::default();
        for meta in args {
            match &meta {
                Meta::NameValue(nv) if nv.path.is_ident("name") => {
                    out.name = Some(expect_str(&nv.value, "name")?);
                }
                Meta::NameValue(nv) if nv.path.is_ident("description") => {
                    out.description = Some(expect_str(&nv.value, "description")?);
                }
                Meta::Path(p) if p.is_ident("needs_approval") => out.needs_approval = true,
                Meta::NameValue(nv) if nv.path.is_ident("needs_approval") => {
                    out.needs_approval = lit_bool(&nv.value).ok_or_else(|| {
                        syn::Error::new_spanned(&nv.value, "expected `true` or `false`")
                    })?;
                }
                Meta::NameValue(MetaNameValue { path, value, .. })
                    if path.is_ident("needs_approval_if") =>
                {
                    let Expr::Path(function) = value else {
                        return Err(syn::Error::new_spanned(
                            value,
                            "`needs_approval_if` expects the path of a `fn(&Self) -> bool`",
                        ));
                    };
                    out.needs_approval_if = Some(function.clone());
                }
                Meta::NameValue(nv) if nv.path.is_ident("timeout") => {
                    out.timeout_seconds = Some(expect_seconds(&nv.value)?);
                }
                Meta::NameValue(nv) if nv.path.is_ident("on_failure") => {
                    let Expr::Lit(lit) = &nv.value else {
                        return Err(on_failure_error(&nv.value));
                    };
                    let Lit::Str(text) = &lit.lit else {
                        return Err(on_failure_error(&nv.value));
                    };
                    if !text.value().contains("{error") {
                        return Err(on_failure_error(&nv.value));
                    }
                    out.on_failure = Some(text.clone());
                }
                Meta::List(list) if list.path.is_ident("requires") => {
                    let types =
                        list.parse_args_with(Punctuated::<Type, Token![,]>::parse_terminated)?;
                    out.requires.extend(types);
                }
                other => {
                    return Err(syn::Error::new_spanned(
                        other,
                        format!(
                            "unsupported `tool` option; supported options: {SUPPORTED_OPTIONS}"
                        ),
                    ));
                }
            }
        }
        Ok(out)
    }
}

fn on_failure_error(spanned: &Expr) -> syn::Error {
    syn::Error::new_spanned(
        spanned,
        "`on_failure` expects a string literal that mentions `{error}`, \
         e.g. `on_failure = \"The refund was not made: {error}\"`",
    )
}

pub(crate) fn expand(input: &DeriveInput) -> syn::Result<TokenStream> {
    let mut attrs = MetaAttrs::default();
    let mut seen = false;
    for attr in input.attrs.iter().filter(|a| a.path().is_ident("tool")) {
        if seen {
            return Err(syn::Error::new_spanned(
                attr,
                "declare one `#[tool(...)]` per struct; merge the options into it",
            ));
        }
        seen = true;
        attrs = MetaAttrs::parse(
            attr.parse_args_with(Punctuated::<Meta, Token![,]>::parse_terminated)?,
        )?;
    }

    let ident = &input.ident;
    let Some(name) = attrs.name else {
        return Err(syn::Error::new_spanned(
            ident,
            "missing `#[tool(name = \"...\")]`: the model needs a name to call the tool by",
        ));
    };
    let Some(description) = attrs.description.or_else(|| crate::doc_text(&input.attrs)) else {
        return Err(syn::Error::new_spanned(
            ident,
            "the tool has no description: write a doc comment on the struct or \
             `#[tool(description = \"...\")]`",
        ));
    };

    let needs_resolution = if let Some(function) = &attrs.needs_approval_if {
        Some(quote! {
            .needs_resolution(opai_agents::NeedsResolution::dynamic(
                |__params, _call_id| async move {
                    // Arguments that do not parse never run, so they never need a person.
                    match opai_agents::serde_json::from_value::<Self>(__params) {
                        Ok(__args) => (#function)(&__args),
                        Err(_) => false,
                    }
                },
            ))
        })
    } else if attrs.needs_approval {
        Some(quote! { .needs_resolution(true) })
    } else {
        None
    };
    let timeout = attrs
        .timeout_seconds
        .map(|seconds| quote! { .timeout(#seconds) });
    let on_failure = attrs.on_failure.map(|text| {
        quote! {
            .on_failure(opai_agents::ToolFailureHandling::Custom(std::sync::Arc::new(
                |_ctx, __error| format!(#text, error = __error),
            )))
        }
    });
    let requires = &attrs.requires;

    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();
    Ok(quote! {
        impl #impl_generics opai_agents::ToolMeta for #ident #ty_generics #where_clause {
            const NAME: &'static str = #name;
            const DESCRIPTION: &'static str = #description;

            fn policy() -> opai_agents::ToolPolicy {
                opai_agents::ToolPolicy::new()
                    #needs_resolution
                    #timeout
                    #on_failure
            }

            fn requires() -> ::std::vec::Vec<opai_agents::TypeRequirement> {
                ::std::vec![ #(opai_agents::TypeRequirement::of::<#requires>()),* ]
            }
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(src: &str) -> syn::Result<MetaAttrs> {
        let args =
            syn::parse::Parser::parse_str(Punctuated::<Meta, Token![,]>::parse_terminated, src)?;
        MetaAttrs::parse(args)
    }

    #[test]
    fn known_options_are_read() {
        let a = parse(
            r#"name = "n", description = "d", needs_approval, timeout = 2.5,
               needs_approval_if = big, on_failure = "no: {error}", requires(A, B)"#,
        )
        .unwrap();
        assert_eq!(a.name.as_deref(), Some("n"));
        assert_eq!(a.description.as_deref(), Some("d"));
        assert!(a.needs_approval);
        assert!(a.needs_approval_if.is_some());
        assert_eq!(a.timeout_seconds, Some(2.5));
        assert_eq!(a.requires.len(), 2);
    }

    #[test]
    fn unknown_and_malformed_options_are_errors() {
        assert!(parse("timeut = 5").is_err());
        assert!(parse("timeout = 0").is_err());
        assert!(parse(r#"on_failure = "no placeholder""#).is_err());
        assert!(parse(r#"needs_approval_if = "big""#).is_err());
        assert!(parse("name = 3").is_err());
    }
}
