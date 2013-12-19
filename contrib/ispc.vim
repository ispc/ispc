" Vim syntax file
" Language:	ISPC
" Maintainer:	Andreas Wendleder <andreas.wendleder@gmail.com>
" Last Change:	2011 Aug 3

" Quit when a syntax file was already loaded
if exists("b:current_syntax")
  finish
endif

" Read the C syntax to start with
runtime! syntax/c.vim
unlet b:current_syntax

" New keywords
syn keyword	ispcStatement	cbreak ccontinue creturn launch print reference soa sync task
syn keyword	ispcConditional	cif
syn keyword	ispcRepeat	cdo cfor cwhile
syn keyword	ispcBuiltin	programCount programIndex	
syn keyword	ispcType	export uniform varying int8 int16 int32 int64
syn keyword	ispcOperator	operator

"double precision floating point number, with dot, optional exponent
syn match	cFloat		display contained "\d\+\.\d*d[-+]\=\d*\>"
"double precision floating point number, starting with dot, optional exponent
syn match	cFloat		display contained ".\d*d[-+]\=\d*\>"
"double precision floating point number, without dot, with exponent
syn match	cFloat		display contained "\d\+d[-+]\=\d\+\>"

" Default highlighting
command -nargs=+ HiLink hi def link <args>
HiLink ispcStatement	Statement
HiLink ispcConditional	Conditional
HiLink ispcRepeat	Repeat
HiLink ispcBuiltin	Statement
HiLink ispcType		Type
HiLink ispcOperator	Operator
delcommand HiLink

let b:current_syntax = "ispc"

