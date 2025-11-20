" Vim syntax file
" Language:		ISPC
" Maintainer:		Dmitry Babokin <dmitry.y.babokin@intel.com>
" Previous Maintainer:	Andreas Wendleder <andreas.wendleder@gmail.com>
" Last Change:		November 21, 2024

" Quit when a syntax file was already loaded
if exists("b:current_syntax")
  finish
endif

" Read the C syntax to start with
runtime! syntax/c.vim
unlet b:current_syntax

" New keywords
syn keyword	ispcStatement		assert assume cbreak ccontinue creturn delete launch new print soa sync task unmasked
syn keyword	ispcConditional		cif
syn keyword	ispcRepeat		cdo cfor cwhile foreach foreach_tiled foreach_unique foreach_active
syn keyword	ispcBuiltin		programCount programIndex taskCount taskCount0 taskCount1 taskCount3 taskIndex taskIndex0 taskIndex1 taskIndex2
syn keyword	ispcType		export uniform varying int8 int16 int32 int64 uint8 uint16 uint32 uint64 float16
syn keyword	ispcOperator		operator in
syn keyword	ispcStorageClass	noinline __vectorcall __regcall
syn keyword	ispcTemplates		template typename
syn keyword	ispcDefine		ISPC ISPC_POINTER_SIZE ISPC_MAJOR_VERSION ISPC_MINOR_VERSION TARGET_WIDTH PI
                                        \ LLVM_VERSION_MAJOR LLVM_VERSION_MINOR
					\ TARGET_ELEMENT_WIDTH ISPC_UINT_IS_DEFINED ISPC_LLVM_INTRINSICS_ENABLED
                                        \ ISPC_TARGET_HAS_FP16_SUPPORT ISPC_TARGET_HAS_FP64_SUPPORT
                                        \ ISPC_ATTRIBUTE_SUPPORTED
                                        \ INT8_MIN INT16_MIN INT32_MIN INT64_MIN INT8_MAX INT16_MAX INT32_MAX INT64_MAX
                                        \ UINT8_MAX UINT16_MAX UINT32_MAX UINT64_MAX
                                        \ F16_MIN FLT_MIN DBL_MIN F16_MAX FLT_MAX DBL_MAX
					\ ISPC_TARGET_NEON ISPC_TARGET_SSE2 ISPC_TARGET_SSE4 ISPC_TARGET_AVX ISPC_TARGET_AVX2 ISPC_TARGET_AVX512KNL ISPC_TARGET_AVX512SKX
					\ ISPC_TARGET_AVX512SPR ISPC_TARGET_AVX10_2
                                        \ ISPC_TARGET_XELP ISPC_TARGET_XEHPG ISPC_TARGET_XEHPC ISPC_TARGET_XELPG ISPC_TARGET_XE2HPG ISPC_TARGET_XE2LPG


" LLVM intrinsics are ISPC intrinsics
syn match	ispcLLVMIntrin	display "@llvm\(\.\w\+\)\+"
" ... operator
syn match	ispcOperator	display "\.\.\."

" Integer literals (in binary, decimal, or hex form), with k/M/G suffix and u/U, l/L, ll/LL suffixes.
syn match	cNumber		display contained "\(\d\+\|0[xX]\x\+\|0[bB][01]\+\)[kMG]\=\([uU]\=\([lL]\|ll\|LL\)\|\([lL]\|ll\|LL\)\=[uU]\)\>"

" Decimal floating point literal with optional suffix
syn match	cFloat		display contained "\(\(\d\+\.\d*\)\|\(\.\d\+\)\)\([fF]16\|[dDfF]\)\=\>"
" [Deprecated] decimal floating point literal with "f" suffix, but " without radix separator
syn match	cFloat		display contained "\d\+[fF]\>"
" Scientific notation floating point literals with optional suffix
syn match	cFloat		display contained "\(\(\d\+\.\d*\)\|\(\.\d\+\)\|\d\+\)[eE][-+]\=\d\+\([fF]16\|[dDfF]\)\=\>"
" Hexadecimal floating point numbers - subset of C++17 hexadecimals, with optional suffix
syn match	cFloat		display contained "0[xX][01]\(.\x\+\)\=[pP][-+]\=\d\+\([fF]16\|[dDfF]\)\=\>"
" "Fortran double format" - same as scientific notation, but with d/D instead of e/E
syn match	cFloat		display contained "\(\(\d\+\.\d*\)\|\(\.\d\+\)\|\d\+\)[dD][-+]\=\d\+\>"

" Default highlighting
command -nargs=+ HiLink hi def link <args>
HiLink ispcStatement	Statement
HiLink ispcLLVMIntrin	Statement
HiLink ispcConditional	Conditional
HiLink ispcRepeat	Repeat
HiLink ispcBuiltin	Statement
HiLink ispcType		Type
HiLink ispcTemplates	Type
HiLink ispcOperator	Operator
HiLink ispcDefine	Define
HiLink ispcStorageClass	StorageClass
delcommand HiLink

let b:current_syntax = "ispc"
