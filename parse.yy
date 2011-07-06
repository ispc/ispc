/*
  Copyright (c) 2010-2011, Intel Corporation
  All rights reserved.

  Redistribution and use in source and binary forms, with or without
  modification, are permitted provided that the following conditions are
  met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    * Neither the name of Intel Corporation nor the names of its
      contributors may be used to endorse or promote products derived from
      this software without specific prior written permission.


   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
   IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
   TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
   PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.  
*/

%locations

/* supress shift-reduces conflict message for dangling else */
/* one for 'if', one for 'uif' */
%expect 2

%pure-parser

%code requires {

#define YYLTYPE SourcePos

# define YYLLOC_DEFAULT(Current, Rhs, N)                               \
    do                                                                 \
      if (N)                                                           \
        {                                                              \
          (Current).first_line   = YYRHSLOC (Rhs, 1).first_line;       \
          (Current).first_column = YYRHSLOC (Rhs, 1).first_column;     \
          (Current).last_line    = YYRHSLOC (Rhs, N).last_line;        \
          (Current).last_column  = YYRHSLOC (Rhs, N).last_column;      \
          (Current).name         = YYRHSLOC (Rhs, 1).name    ;         \
        }                                                              \
      else                                                             \
        { /* empty RHS */                                              \
          (Current).first_line   = (Current).last_line   =             \
            YYRHSLOC (Rhs, 0).last_line;                               \
          (Current).first_column = (Current).last_column =             \
            YYRHSLOC (Rhs, 0).last_column;                             \
          (Current).name = NULL;                        /* new */ \
        }                                                              \
    while (0)
}

%{

#include "ispc.h"
#include "type.h"
#include "module.h"
#include "decl.h"
#include "expr.h"
#include "sym.h"
#include "stmt.h"
#include "util.h"

#include <stdio.h>
#include <llvm/Constants.h>

#define UNIMPLEMENTED \
        Error(yylloc, "Unimplemented parser functionality %s:%d", \
        __FILE__, __LINE__);

union YYSTYPE;
extern int yylex(YYSTYPE *, SourcePos *);

extern char *yytext;

void yyerror(const char *s) { fprintf(stderr, "Parse error: %s\n", s); }

static void lAddFunctionParams(Declarator *decl);
static void lAddMaskToSymbolTable(SourcePos pos);
static void lAddThreadIndexCountToSymbolTable(SourcePos pos);
static std::string lGetAlternates(std::vector<std::string> &alternates);
static const char *lGetStorageClassString(StorageClass sc);

static const char *lBuiltinTokens[] = {
    "bool", "break", "case", "cbreak", "ccontinue", "cdo", "cfor", "char", 
    "cif", "cwhile", "const", "continue", "creturn", "default", "do", "double", 
    "else", "enum", "export", "extern", "false", "float", "for", "goto", "if",
    "inline", "int", "int32", "int64", "launch", "print", "reference", "return",
    "static", "struct", "switch", "sync", "task", "true", "typedef", "uniform",
    "unsigned", "varying", "void", "while", NULL 
};
    
%}

%union {
    Expr *expr;
    ExprList *exprList;
    const Type *type;
    const AtomicType *atomicType;
    int typeQualifier;
    StorageClass storageClass;
    Stmt *stmt;
    DeclSpecs *declSpecs;
    Declaration *declaration;
    std::vector<Declarator *> *declarators;
    std::vector<Declaration *> *declarationList;
    Declarator *declarator;
    std::vector<Declarator *> *structDeclaratorList;
    StructDeclaration *structDeclaration;
    std::vector<StructDeclaration *> *structDeclarationList;
    int32_t int32Val;
    uint32_t uint32Val;
    double floatVal;
    int64_t int64Val;
    uint64_t uint64Val;
    std::string *stringVal;
    const char *constCharPtr;
}


%token TOKEN_IDENTIFIER TOKEN_INT_CONSTANT TOKEN_UINT_CONSTANT TOKEN_FLOAT_CONSTANT 
%token TOKEN_STRING_LITERAL TOKEN_TYPE_NAME
%token TOKEN_PTR_OP TOKEN_INC_OP TOKEN_DEC_OP TOKEN_LEFT_OP TOKEN_RIGHT_OP 
%token TOKEN_LE_OP TOKEN_GE_OP TOKEN_EQ_OP TOKEN_NE_OP
%token TOKEN_AND_OP TOKEN_OR_OP TOKEN_MUL_ASSIGN TOKEN_DIV_ASSIGN TOKEN_MOD_ASSIGN 
%token TOKEN_ADD_ASSIGN TOKEN_SUB_ASSIGN TOKEN_LEFT_ASSIGN TOKEN_RIGHT_ASSIGN 
%token TOKEN_AND_ASSIGN TOKEN_OR_ASSIGN TOKEN_XOR_ASSIGN

%token TOKEN_EXTERN TOKEN_EXPORT TOKEN_STATIC TOKEN_INLINE TOKEN_TASK 
%token TOKEN_UNIFORM TOKEN_VARYING TOKEN_TYPEDEF TOKEN_SOA
%token TOKEN_CHAR TOKEN_INT TOKEN_UNSIGNED TOKEN_FLOAT TOKEN_DOUBLE
%token TOKEN_INT64 TOKEN_CONST TOKEN_VOID TOKEN_BOOL 
%token TOKEN_ENUM TOKEN_STRUCT TOKEN_TRUE TOKEN_FALSE TOKEN_REFERENCE

%token TOKEN_CASE TOKEN_DEFAULT TOKEN_IF TOKEN_ELSE TOKEN_SWITCH
%token TOKEN_WHILE TOKEN_DO TOKEN_LAUNCH
%token TOKEN_FOR TOKEN_GOTO TOKEN_CONTINUE TOKEN_BREAK TOKEN_RETURN
%token TOKEN_CIF TOKEN_CDO TOKEN_CFOR TOKEN_CWHILE
%token TOKEN_CBREAK TOKEN_CCONTINUE TOKEN_CRETURN TOKEN_SYNC TOKEN_PRINT

%type <expr> primary_expression postfix_expression
%type <expr> unary_expression cast_expression
%type <expr> multiplicative_expression additive_expression shift_expression
%type <expr> relational_expression equality_expression and_expression
%type <expr> exclusive_or_expression inclusive_or_expression
%type <expr> logical_and_expression logical_or_expression 
%type <expr> conditional_expression assignment_expression expression
%type <expr> initializer constant_expression for_test
%type <exprList> argument_expression_list initializer_list

%type <stmt> statement labeled_statement compound_statement for_init_statement
%type <stmt> expression_statement selection_statement iteration_statement
%type <stmt> jump_statement statement_list declaration_statement print_statement

%type <declaration> declaration parameter_declaration
%type <declarators> init_declarator_list 
%type <declarationList> parameter_list parameter_type_list
%type <declarator> declarator init_declarator direct_declarator struct_declarator

%type <structDeclaratorList> struct_declarator_list
%type <structDeclaration> struct_declaration
%type <structDeclarationList> struct_declaration_list

%type <type> specifier_qualifier_list struct_or_union_specifier
%type <type> enum_specifier type_specifier type_name
%type <type> short_vec_specifier
%type <atomicType> atomic_var_type_specifier

%type <typeQualifier> type_qualifier
%type <storageClass> storage_class_specifier
%type <declSpecs> declaration_specifiers 

%type <stringVal> string_constant
%type <constCharPtr> struct_or_union_name
%type <int32Val> int_constant soa_width_specifier

%start translation_unit
%%

string_constant
    : TOKEN_STRING_LITERAL { $$ = new std::string(*yylval.stringVal); }
    ;

primary_expression
    : TOKEN_IDENTIFIER {
        const char *name = yylval.stringVal->c_str();
        Symbol *s = m->symbolTable->LookupVariable(name);
        $$ = NULL;
        if (s)
            $$ = new SymbolExpr(s, @1);       
        else {
            std::vector<Symbol *> *funs = m->symbolTable->LookupFunction(name);
            if (funs)
                $$ = new FunctionSymbolExpr(funs, @1);
        }
        if ($$ == NULL) {
            std::vector<std::string> alternates = 
                m->symbolTable->ClosestVariableOrFunctionMatch(name);
            std::string alts = lGetAlternates(alternates);
            Error(@1, "Undeclared symbol \"%s\".%s", name, alts.c_str());
        }
    }
    | TOKEN_INT_CONSTANT {
        /* FIXME: should support 64 bit constants (and doubles...) */
        $$ = new ConstExpr(AtomicType::UniformConstInt32, yylval.int32Val, @1); 
    }
    | TOKEN_UINT_CONSTANT {
        $$ = new ConstExpr(AtomicType::UniformConstUInt32, yylval.uint32Val, @1); 
    }
    | TOKEN_FLOAT_CONSTANT {
        $$ = new ConstExpr(AtomicType::UniformConstFloat, (float)yylval.floatVal, @1); 
    }
    | TOKEN_TRUE {
        $$ = new ConstExpr(AtomicType::UniformConstBool, true, @1);
    }
    | TOKEN_FALSE {
        $$ = new ConstExpr(AtomicType::UniformConstBool, false, @1);
    }
/*    | TOKEN_STRING_LITERAL
       { UNIMPLEMENTED }*/
    | '(' expression ')' { $$ = $2; }
    ;

postfix_expression
    : primary_expression
    | postfix_expression '[' expression ']'
      { $$ = new IndexExpr($1, $3, @1); }
    | postfix_expression '(' ')'
      { $$ = new FunctionCallExpr($1, new ExprList(@1), @1, false); }
    | postfix_expression '(' argument_expression_list ')'
      { $$ = new FunctionCallExpr($1, $3, @1, false); }
    | TOKEN_LAUNCH '<' postfix_expression '(' argument_expression_list ')' '>'
      { $$ = new FunctionCallExpr($3, $5, @3, true); }
    | TOKEN_LAUNCH '<' postfix_expression '(' ')' '>'
      { $$ = new FunctionCallExpr($3, new ExprList(@3), @3, true); }
    | postfix_expression '.' TOKEN_IDENTIFIER
      { $$ = new MemberExpr($1, yytext, @1, @3); }
/*    | postfix_expression TOKEN_PTR_OP TOKEN_IDENTIFIER
      { UNIMPLEMENTED }
*/
    | postfix_expression TOKEN_INC_OP
      { $$ = new UnaryExpr(UnaryExpr::PostInc, $1, @1); }
    | postfix_expression TOKEN_DEC_OP
      { $$ = new UnaryExpr(UnaryExpr::PostDec, $1, @1); }
    ;

argument_expression_list
    : assignment_expression      { $$ = new ExprList($1, @1); }
    | argument_expression_list ',' assignment_expression
      {
          ExprList *argList = dynamic_cast<ExprList *>($1);
          assert(argList != NULL);
          argList->exprs.push_back($3);
          $$ = argList;
      }
    ;

unary_expression
    : postfix_expression
    | TOKEN_INC_OP unary_expression   
      { $$ = new UnaryExpr(UnaryExpr::PreInc, $2, @2); }
    | TOKEN_DEC_OP unary_expression   
      { $$ = new UnaryExpr(UnaryExpr::PreDec, $2, @2); }
    | '+' cast_expression 
      { $$ = $2; }
    | '-' cast_expression 
      { $$ = new UnaryExpr(UnaryExpr::Negate, $2, @2); }
    | '~' cast_expression 
      { $$ = new UnaryExpr(UnaryExpr::BitNot, $2, @2); }
    | '!' cast_expression 
      { $$ = new UnaryExpr(UnaryExpr::LogicalNot, $2, @2); }
    ;

cast_expression
    : unary_expression
    | '(' type_name ')' cast_expression
      {
          // If type_name isn't explicitly a varying, We do a GetUniform() 
          // call here so that things like:
          // uniform int y = ...;
          // uniform float x = 1. / (float)y;
          // don't issue an error due to (float)y being inadvertently
          // and undesirably-to-the-user "varying"...
          if ($4->GetType()->IsUniformType())
             $2 = $2->GetAsUniformType();
          $$ = new TypeCastExpr($2, $4, @1); 
      }
    ;

multiplicative_expression
    : cast_expression
    | multiplicative_expression '*' cast_expression
      { $$ = new BinaryExpr(BinaryExpr::Mul, $1, $3, @2); }
    | multiplicative_expression '/' cast_expression
      { $$ = new BinaryExpr(BinaryExpr::Div, $1, $3, @2); }
    | multiplicative_expression '%' cast_expression
      { $$ = new BinaryExpr(BinaryExpr::Mod, $1, $3, @2); }
    ;

additive_expression
    : multiplicative_expression
    | additive_expression '+' multiplicative_expression
      { $$ = new BinaryExpr(BinaryExpr::Add, $1, $3, @2); }
    | additive_expression '-' multiplicative_expression
      { $$ = new BinaryExpr(BinaryExpr::Sub, $1, $3, @2); }
    ;

shift_expression
    : additive_expression
    | shift_expression TOKEN_LEFT_OP additive_expression
      { $$ = new BinaryExpr(BinaryExpr::Shl, $1, $3, @1); }
    | shift_expression TOKEN_RIGHT_OP additive_expression
      { $$ = new BinaryExpr(BinaryExpr::Shr, $1, $3, @1); }
    ;

relational_expression
    : shift_expression
    | relational_expression '<' shift_expression
      { $$ = new BinaryExpr(BinaryExpr::Lt, $1, $3, @2); }
    | relational_expression '>' shift_expression
      { $$ = new BinaryExpr(BinaryExpr::Gt, $1, $3, @2); }
    | relational_expression TOKEN_LE_OP shift_expression
      { $$ = new BinaryExpr(BinaryExpr::Le, $1, $3, @2); }
    | relational_expression TOKEN_GE_OP shift_expression
      { $$ = new BinaryExpr(BinaryExpr::Ge, $1, $3, @2); }
    ;

equality_expression
    : relational_expression
    | equality_expression TOKEN_EQ_OP relational_expression
      { $$ = new BinaryExpr(BinaryExpr::Equal, $1, $3, @2); }
    | equality_expression TOKEN_NE_OP relational_expression
      { $$ = new BinaryExpr(BinaryExpr::NotEqual, $1, $3, @2); }
    ;

and_expression
    : equality_expression
    | and_expression '&' equality_expression
      { $$ = new BinaryExpr(BinaryExpr::BitAnd, $1, $3, @2); }
    ;

exclusive_or_expression
    : and_expression
    | exclusive_or_expression '^' and_expression
      { $$ = new BinaryExpr(BinaryExpr::BitXor, $1, $3, @2); }
    ;

inclusive_or_expression
    : exclusive_or_expression
    | inclusive_or_expression '|' exclusive_or_expression
      { $$ = new BinaryExpr(BinaryExpr::BitOr, $1, $3, @2); }
    ;

logical_and_expression
    : inclusive_or_expression
    | logical_and_expression TOKEN_AND_OP inclusive_or_expression
      { $$ = new BinaryExpr(BinaryExpr::LogicalAnd, $1, $3, @2); }
    ;

logical_or_expression
    : logical_and_expression
    | logical_or_expression TOKEN_OR_OP logical_and_expression
      { $$ = new BinaryExpr(BinaryExpr::LogicalOr, $1, $3, @2); }
    ;

conditional_expression
    : logical_or_expression
    | logical_or_expression '?' expression ':' conditional_expression
      { $$ = new SelectExpr($1, $3, $5, @1); }
    ;

assignment_expression
    : conditional_expression
    | unary_expression '=' assignment_expression
      { $$ = new AssignExpr(AssignExpr::Assign, $1, $3, @2); }
    | unary_expression TOKEN_MUL_ASSIGN assignment_expression
      { $$ = new AssignExpr(AssignExpr::MulAssign, $1, $3, @2); }
    | unary_expression TOKEN_DIV_ASSIGN assignment_expression
      { $$ = new AssignExpr(AssignExpr::DivAssign, $1, $3, @2); }
    | unary_expression TOKEN_MOD_ASSIGN assignment_expression
      { $$ = new AssignExpr(AssignExpr::ModAssign, $1, $3, @2); }
    | unary_expression TOKEN_ADD_ASSIGN assignment_expression
      { $$ = new AssignExpr(AssignExpr::AddAssign, $1, $3, @2); }
    | unary_expression TOKEN_SUB_ASSIGN assignment_expression
      { $$ = new AssignExpr(AssignExpr::SubAssign, $1, $3, @2); }
    | unary_expression TOKEN_LEFT_ASSIGN assignment_expression
      { $$ = new AssignExpr(AssignExpr::ShlAssign, $1, $3, @2); }
    | unary_expression TOKEN_RIGHT_ASSIGN assignment_expression
      { $$ = new AssignExpr(AssignExpr::ShrAssign, $1, $3, @2); }
    | unary_expression TOKEN_AND_ASSIGN assignment_expression
      { $$ = new AssignExpr(AssignExpr::AndAssign, $1, $3, @2); }
    | unary_expression TOKEN_XOR_ASSIGN assignment_expression
      { $$ = new AssignExpr(AssignExpr::XorAssign, $1, $3, @2); }
    | unary_expression TOKEN_OR_ASSIGN assignment_expression
      { $$ = new AssignExpr(AssignExpr::OrAssign, $1, $3, @2); }
    ;

expression
    : assignment_expression
    | TOKEN_SYNC 
      { $$ = new SyncExpr(@1); }
    | expression ',' assignment_expression
      { $$ = new BinaryExpr(BinaryExpr::Comma, $1, $3, @2); }
    ;

constant_expression
    : conditional_expression
    ;

declaration_statement
    : declaration     { $$ = new DeclStmt(@1, $1, m->symbolTable); }
    ;

declaration
    : declaration_specifiers ';'
      {
          $$ = new Declaration($1);
      }
    | declaration_specifiers init_declarator_list ';'
      {
          $$ = new Declaration($1, $2);
      }
    ;

soa_width_specifier
    : TOKEN_SOA '<' int_constant '>'
      { $$ = $3; }
    ;

declaration_specifiers
    : storage_class_specifier
      {
          $$ = new DeclSpecs(NULL, $1);
      }
    | storage_class_specifier declaration_specifiers
      {
          DeclSpecs *ds = (DeclSpecs *)$2;
          if (ds->storageClass != SC_NONE)
              Error(@1, "Multiple storage class specifiers in a declaration are illegal. "
                    "(Have provided both \"%s\" and \"%s\".)",
                    lGetStorageClassString(ds->storageClass),
                    lGetStorageClassString($1));
          else
              ds->storageClass = $1;
          $$ = ds;
      }
    | soa_width_specifier
      {
          DeclSpecs *ds = new DeclSpecs;
          ds->soaWidth = $1;
          $$ = ds;
      }
    | soa_width_specifier declaration_specifiers
      {
          DeclSpecs *ds = (DeclSpecs *)$2;
          if (ds->soaWidth != 0)
              Error(@1, "soa<> qualifier supplied multiple times in declaration.");
          else
              ds->soaWidth = $1;
          $$ = ds;
      }
    | type_specifier
      {
          $$ = new DeclSpecs($1);
      }
    | type_specifier '<' int_constant '>'
    {
          DeclSpecs *ds = new DeclSpecs($1);
          ds->vectorSize = $3;
          $$ = ds;
    }
    | type_specifier declaration_specifiers
      {
          DeclSpecs *ds = (DeclSpecs *)$2;
          if (ds->baseType != NULL)
              Error(@1, "Multiple types provided for declaration.");
          ds->baseType = $1;
          $$ = ds;
      }
    | type_qualifier
      {
          $$ = new DeclSpecs(NULL, SC_NONE, $1);
      }
    | type_qualifier declaration_specifiers
      {
          DeclSpecs *ds = (DeclSpecs *)$2;
          ds->typeQualifier |= $1;
          $$ = ds;
      }
    ;

init_declarator_list
    : init_declarator
      {
          std::vector<Declarator *> *dl = new std::vector<Declarator *>;
          dl->push_back($1);
          $$ = dl;
      }
    | init_declarator_list ',' init_declarator
      {
          std::vector<Declarator *> *dl = (std::vector<Declarator *> *)$1;
          dl->push_back($3);
          $$ = $1;
      }
    ;

init_declarator
    : declarator
    | declarator '=' initializer { $1->initExpr = $3; $$ = $1; }
    ;

storage_class_specifier
    : TOKEN_TYPEDEF { $$ = SC_TYPEDEF; }
    | TOKEN_EXTERN { $$ = SC_EXTERN; }
    | TOKEN_EXTERN TOKEN_STRING_LITERAL  { 
          if (*(yylval.stringVal) != "C")
              Error(@2, "String constant \"%s\" illegal after \"extern\" qualifier.  "
                    "(Expected \"C\".)", yylval.stringVal->c_str());
          $$ = SC_EXTERN_C; 
      }
    | TOKEN_EXPORT { $$ = SC_EXPORT; }
    | TOKEN_STATIC { $$ = SC_STATIC; }
    ;

type_specifier
    : atomic_var_type_specifier { $$ = $1; }
    | TOKEN_TYPE_NAME
      { const Type *t = m->symbolTable->LookupType(yytext); 
        assert(t != NULL);
        $$ = t;
      }
    | struct_or_union_specifier { $$ = $1; }
    | enum_specifier 
      { UNIMPLEMENTED; }
/*    | TOKEN_TYPE_NAME 
      { UNIMPLEMENTED; }
*/
    ;

atomic_var_type_specifier
    : TOKEN_VOID { $$ = AtomicType::Void; }
    | TOKEN_BOOL { $$ = AtomicType::VaryingBool; }
/*  | TOKEN_CHAR { UNIMPLEMENTED; } */
    | TOKEN_INT { $$ = AtomicType::VaryingInt32; }
    | TOKEN_FLOAT { $$ = AtomicType::VaryingFloat; }
    | TOKEN_DOUBLE { $$ = AtomicType::VaryingDouble; }
    | TOKEN_INT64 { $$ = AtomicType::VaryingInt64; }
    ;

short_vec_specifier
    : atomic_var_type_specifier '<' int_constant '>'
      {
        Type* vt = 
          new VectorType($1, $3);
        $$ = vt;
      }
    ;

struct_or_union_name
    : TOKEN_IDENTIFIER { $$ = strdup(yytext); }
    | TOKEN_TYPE_NAME { $$ = strdup(yytext); }
    ;

struct_or_union_specifier
    : struct_or_union struct_or_union_name '{' struct_declaration_list '}' 
      { 
          std::vector<const Type *> elementTypes;
          std::vector<std::string> elementNames;
          std::vector<SourcePos> elementPositions;
          GetStructTypesNamesPositions(*$4, &elementTypes, &elementNames,
                                       &elementPositions);
          StructType *st = new StructType($2, elementTypes, elementNames,
                                          elementPositions, false, true, @2);
          m->symbolTable->AddType($2, st, @2);
          $$ = st;
      }
    | struct_or_union '{' struct_declaration_list '}' 
      {
          std::vector<const Type *> elementTypes;
          std::vector<std::string> elementNames;
          std::vector<SourcePos> elementPositions;
          GetStructTypesNamesPositions(*$3, &elementTypes, &elementNames,
                                       &elementPositions);
          $$ = new StructType("", elementTypes, elementNames, elementPositions,
                              false, true, @1);
      }
    | struct_or_union '{' '}' 
      {
          Error(@1, "Empty struct definitions not allowed."); 
      }
    | struct_or_union struct_or_union_name '{' '}' 
      {
          Error(@1, "Empty struct definitions not allowed."); 
      }
    | struct_or_union struct_or_union_name
      { const Type *st = m->symbolTable->LookupType($2); 
        if (!st) {
            std::vector<std::string> alternates = m->symbolTable->ClosestTypeMatch($2);
            std::string alts = lGetAlternates(alternates);
            Error(@2, "Struct type \"%s\" unknown.%s", $2, alts.c_str());
        }
        else if (dynamic_cast<const StructType *>(st) == NULL)
            Error(@2, "Type \"%s\" is not a struct type!", $2);
        $$ = st;
      }
    ;

struct_or_union
    : TOKEN_STRUCT 
    ;

struct_declaration_list
    : struct_declaration 
      { 
          std::vector<StructDeclaration *> *sdl = new std::vector<StructDeclaration *>;
          sdl->push_back($1);
          $$ = sdl;
      }
    | struct_declaration_list struct_declaration 
      {
          std::vector<StructDeclaration *> *sdl = (std::vector<StructDeclaration *> *)$1;
          sdl->push_back($2);
          $$ = $1;
      }
    ;

struct_declaration
    : specifier_qualifier_list struct_declarator_list ';' 
      { $$ = new StructDeclaration($1, $2); }
    ;

specifier_qualifier_list
    : type_specifier specifier_qualifier_list
    | type_specifier
    | short_vec_specifier
    | type_qualifier specifier_qualifier_list 
    {
        if ($1 == TYPEQUAL_UNIFORM)
            $$ = $2->GetAsUniformType();
        else if ($1 == TYPEQUAL_VARYING)
            $$ = $2->GetAsVaryingType();
        else if ($1 == TYPEQUAL_REFERENCE)
            $$ = new ReferenceType($2, false);
        else if ($1 == TYPEQUAL_CONST)
            $$ = $2->GetAsConstType();
        else if ($1 == TYPEQUAL_UNSIGNED) {
            const Type *t = $2->GetAsUnsignedType();
            if (t)
                $$ = t;
            else {
                Error(@1, "Can't apply \"unsigned\" qualifier to \"%s\" type. Ignoring.",
                      $2->GetString().c_str());
                $$ = $2;
            }
        } 
        else {
            UNIMPLEMENTED;
        }
    }
/* K&R--implicit int type--e.g. "static foo" -> foo is an int */
/*    | type_qualifier { UNIMPLEMENTED; }*/
    ;


struct_declarator_list
    : struct_declarator 
      {
          std::vector<Declarator *> *sdl = new std::vector<Declarator *>;
          sdl->push_back($1);
          $$ = sdl;
      }
    | struct_declarator_list ',' struct_declarator 
      {
          std::vector<Declarator *> *sdl = (std::vector<Declarator *> *)$1;
          sdl->push_back($3);
          $$ = $1;
      }
    ;

struct_declarator
    : declarator { $$ = $1; }
/* bitfields
    | ':' constant_expression
    | declarator ':' constant_expression
*/
    ;

enum_specifier
    : TOKEN_ENUM '{' enumerator_list '}' 
      { UNIMPLEMENTED; }
    | TOKEN_ENUM TOKEN_IDENTIFIER '{' enumerator_list '}' 
      { UNIMPLEMENTED; }
    | TOKEN_ENUM TOKEN_IDENTIFIER 
      { UNIMPLEMENTED; }
    ;

enumerator_list
    : enumerator 
      { UNIMPLEMENTED; }
    | enumerator_list ',' enumerator
    ;

enumerator
    : TOKEN_IDENTIFIER
    | TOKEN_IDENTIFIER '=' constant_expression
    ;

type_qualifier
    : TOKEN_CONST      { $$ = TYPEQUAL_CONST; }
    | TOKEN_UNIFORM    { $$ = TYPEQUAL_UNIFORM; }
    | TOKEN_VARYING    { $$ = TYPEQUAL_VARYING; }
    | TOKEN_TASK       { $$ = TYPEQUAL_TASK; }
    | TOKEN_INLINE     { $$ = TYPEQUAL_INLINE; }
    | TOKEN_REFERENCE  { $$ = TYPEQUAL_REFERENCE; }
    | TOKEN_UNSIGNED   { $$ = TYPEQUAL_UNSIGNED; }
    ;

declarator
    : direct_declarator
    ;

int_constant
    : TOKEN_INT_CONSTANT { $$ = yylval.int32Val; }
    ;

direct_declarator
    : TOKEN_IDENTIFIER
      {
          Symbol *sym = new Symbol(yytext, @1);
          $$ = new Declarator(sym, @1);
      }
    | '(' declarator ')' { $$ = $2; }
    | direct_declarator '[' constant_expression ']'
    {
        Expr *size = $3;
        if (size) size = size->TypeCheck();
        if (size) {
            size = size->Optimize();
            llvm::Constant *cval = size->GetConstant(size->GetType());
            if (!cval) {
                Error(@3, "Array dimension must be compile-time constant");
                $$ = NULL;
            }
            else {
                llvm::ConstantInt *ci = llvm::dyn_cast<llvm::ConstantInt>(cval);
                if (!ci) {
                    Error(@3, "Array dimension must be compile-time integer constant.");
                    $$ = NULL;
                }
                $1->AddArrayDimension((int)ci->getZExtValue());
                $$ = $1;
            }
        }
        else
            $$ = NULL;
    }
    | direct_declarator '[' ']'
    {
        $1->AddArrayDimension(-1); // unsized
        $$ = $1;
    }
    | direct_declarator '(' parameter_type_list ')'
      {
          Declarator *d = (Declarator *)$1;
          d->isFunction = true;
          d->functionArgs = $3;
          $$ = d;
      }
/* K&R?   | direct_declarator '(' identifier_list ')' */
    | direct_declarator '(' ')'
      {
          Declarator *d = (Declarator *)$1;
          d->isFunction = true;
          $$ = d;
      }
    ;


parameter_type_list
    : parameter_list { $$ = $1; }
    ;

parameter_list
    : parameter_declaration
    {
        std::vector<Declaration *> *dl = new std::vector<Declaration *>;
        dl->push_back($1);
        $$ = dl;
    }
    | parameter_list ',' parameter_declaration
    {
        std::vector<Declaration *> *dl = (std::vector<Declaration *> *)$1;
        dl->push_back($3);
        $$ = dl;
    }
    ;

parameter_declaration
    : declaration_specifiers declarator
    {
        $$ = new Declaration($1, $2); 
    }
    | declaration_specifiers declarator '=' initializer
    { 
        $2->initExpr = $4;
        $$ = new Declaration($1, $2); 

    }
    | declaration_specifiers abstract_declarator
    {
        UNIMPLEMENTED;
    }
    | declaration_specifiers
    {
        $$ = new Declaration($1); 
    }
    ;

/* K&R? 
identifier_list
    : IDENTIFIER
    | identifier_list ',' IDENTIFIER
    ;
*/

type_name
    : specifier_qualifier_list
    | specifier_qualifier_list abstract_declarator
    ;

abstract_declarator
/*    : pointer
    | direct_abstract_declarator
    | pointer direct_abstract_declarator */
    : direct_abstract_declarator { UNIMPLEMENTED; }
    ;

direct_abstract_declarator
    : '(' abstract_declarator ')'
/*     | '[' ']' */
    | '[' constant_expression ']'
/*    | direct_abstract_declarator '[' ']' */
    | direct_abstract_declarator '[' constant_expression ']'
    | '(' ')'
    | '(' parameter_type_list ')'
    | direct_abstract_declarator '(' ')'
    | direct_abstract_declarator '(' parameter_type_list ')'
    ;

initializer
    : assignment_expression
    | '{' initializer_list '}' { $$ = $2; }
    | '{' initializer_list ',' '}' { $$ = $2; }
    ;

initializer_list
    : initializer
      { $$ = new ExprList($1, @1); }
    | initializer_list ',' initializer
      {
          ExprList *exprList = dynamic_cast<ExprList *>($1);
          assert(exprList);
          exprList->exprs.push_back($3);
          $$ = exprList;
      }
    ;

statement
    : labeled_statement
    | compound_statement
    | expression_statement
    | selection_statement
    | iteration_statement
    | jump_statement
    | declaration_statement
    | print_statement
    ;

labeled_statement
    : TOKEN_CASE constant_expression ':' statement
      { UNIMPLEMENTED; }
    | TOKEN_DEFAULT ':' statement
      { UNIMPLEMENTED; }
    ;

start_scope
    : '{' { m->symbolTable->PushScope(); }
    ;

end_scope
    : '}' { m->symbolTable->PopScope(); }
    ;

compound_statement
    : '{' '}' { $$ = NULL; }
    | start_scope statement_list end_scope { $$ = $2; }
    ;

statement_list
    : statement
      {
          StmtList *sl = new StmtList(@1);
          sl->Add($1);
          $$ = sl;
      }
    | statement_list statement
      {
          ((StmtList *)$1)->Add($2);
          $$ = $1;
      }
    ;

expression_statement
    : ';' { $$ = NULL; }
    | expression ';' { $$ = new ExprStmt($1, @1); }
    ;

selection_statement
    : TOKEN_IF '(' expression ')' statement
      { $$ = new IfStmt($3, $5, NULL, false, @1); }
    | TOKEN_IF '(' expression ')' statement TOKEN_ELSE statement
      { $$ = new IfStmt($3, $5, $7, false, @1); }
    | TOKEN_CIF '(' expression ')' statement
      { $$ = new IfStmt($3, $5, NULL, true, @1); }
    | TOKEN_CIF '(' expression ')' statement TOKEN_ELSE statement
      { $$ = new IfStmt($3, $5, $7, true, @1); }
    | TOKEN_SWITCH '(' expression ')' statement
      { UNIMPLEMENTED; }
    ;

for_test
    : ';'
      { $$ = NULL; }
    | expression ';'
      { $$ = $1; }
    ;

for_init_statement
    : expression_statement
    | declaration_statement
    ;

for_scope
    : TOKEN_FOR { m->symbolTable->PushScope(); }
    ;

cfor_scope
    : TOKEN_CFOR { m->symbolTable->PushScope(); }
    ;

iteration_statement
    : TOKEN_WHILE '(' expression ')' statement
      { $$ = new ForStmt(NULL, $3, NULL, $5, false, @1); }
    | TOKEN_CWHILE '(' expression ')' statement
      { $$ = new ForStmt(NULL, $3, NULL, $5, true, @1); }
    | TOKEN_DO statement TOKEN_WHILE '(' expression ')' ';'
      { $$ = new DoStmt($5, $2, false, @1); }
    | TOKEN_CDO statement TOKEN_WHILE '(' expression ')' ';'
      { $$ = new DoStmt($5, $2, true, @1); }
    | for_scope '(' for_init_statement for_test ')' statement
      { $$ = new ForStmt($3, $4, NULL, $6, false, @1); 
        m->symbolTable->PopScope();
      }
    | for_scope '(' for_init_statement for_test expression ')' statement
      { $$ = new ForStmt($3, $4, new ExprStmt($5, @5), $7, false, @1); 
        m->symbolTable->PopScope();
      }
    | cfor_scope '(' for_init_statement for_test ')' statement
      { $$ = new ForStmt($3, $4, NULL, $6, true, @1);
        m->symbolTable->PopScope();
      }
    | cfor_scope '(' for_init_statement for_test expression ')' statement
      { $$ = new ForStmt($3, $4, new ExprStmt($5, @5), $7, true, @1);
        m->symbolTable->PopScope();
      }
    ;

jump_statement
    : TOKEN_GOTO TOKEN_IDENTIFIER ';'
      { UNIMPLEMENTED; }
    | TOKEN_CONTINUE ';'
      { $$ = new ContinueStmt(false, @1); }
    | TOKEN_BREAK ';'
      { $$ = new BreakStmt(false, @1); }
    | TOKEN_RETURN ';'
      { $$ = new ReturnStmt(NULL, false, @1); }
    | TOKEN_RETURN expression ';'
      { $$ = new ReturnStmt($2, false, @1); }
    | TOKEN_CCONTINUE ';'
      { $$ = new ContinueStmt(true, @1); }
    | TOKEN_CBREAK ';'
      { $$ = new BreakStmt(true, @1); }
    | TOKEN_CRETURN ';'
      { $$ = new ReturnStmt(NULL, true, @1); }
    | TOKEN_CRETURN expression ';'
      { $$ = new ReturnStmt($2, true, @1); }
    ;

print_statement
    : TOKEN_PRINT '(' string_constant ')'
      {
           $$ = new PrintStmt(*$3, NULL, @1); 
      }
    | TOKEN_PRINT '(' string_constant ',' argument_expression_list ')'
      {
           $$ = new PrintStmt(*$3, $5, @1); 
      }
    ;

translation_unit
    : external_declaration
    | translation_unit external_declaration
    | error
    {
        std::vector<std::string> builtinTokens;
        const char **token = lBuiltinTokens;
        while (*token) {
            builtinTokens.push_back(*token);
            ++token;
        }
        std::vector<std::string> alternates = MatchStrings(yytext, builtinTokens);
        std::string alts = lGetAlternates(alternates);
        Error(@1, "Syntax error--token \"%s\" unknown.%s", yytext, alts.c_str());
    }
    ;

external_declaration
    : function_definition
    | TOKEN_EXTERN TOKEN_STRING_LITERAL '{' declaration '}' // FIXME: make sure string=="C"
    | declaration 
    { 
        for (unsigned int i = 0; i < $1->declarators.size(); ++i)
            m->AddGlobal($1->declSpecs, $1->declarators[i]);
    }
    ;

function_definition
    : declaration_specifiers declarator 
    {
        m->AddGlobal($1, $2);
        lAddFunctionParams($2); 
        lAddMaskToSymbolTable(@2);
        if ($1->typeQualifier & TYPEQUAL_TASK)
            lAddThreadIndexCountToSymbolTable(@2);
    } 
    compound_statement
    {
        m->AddFunction($1, $2, $4);
        m->symbolTable->PopScope(); // push in lAddFunctionParams();
    }
/* function with no declared return type??
func(...) 
    | declarator { lAddFunctionParams($1); } compound_statement
    {
        m->AddFunction(new DeclSpecs(, $1, $3);
        m->symbolTable->PopScope(); // push in lAddFunctionParams();
    }
*/
    ;

%%


/** We're about to start parsing the body of a function; add all of the
    parameters to the symbol table so that they're available.
*/
static void
lAddFunctionParams(Declarator *decl) {
    m->symbolTable->PushScope();

    // wire up arguments
    if (decl->functionArgs) {
        for (unsigned int i = 0; i < decl->functionArgs->size(); ++i) {
            Declaration *pdecl = (*decl->functionArgs)[i];
            assert(pdecl->declarators.size() == 1);
            Symbol *sym = pdecl->declarators[0]->sym;
#ifndef NDEBUG
            bool ok = m->symbolTable->AddVariable(sym);
            assert(ok); // or error message?
#else
            m->symbolTable->AddVariable(sym);
#endif
        }
    }

    // The corresponding pop scope happens in function_definition rules
    // above...
}


/** Add a symbol for the built-in mask variable to the symbol table */
static void lAddMaskToSymbolTable(SourcePos pos) {
    const Type *t = AtomicType::VaryingConstUInt32;
    Symbol *maskSymbol = new Symbol("__mask", pos, t);
    m->symbolTable->AddVariable(maskSymbol);
}


/** Add the thread index and thread count variables to the symbol table
    (this should only be done for 'task'-qualified functions. */
static void lAddThreadIndexCountToSymbolTable(SourcePos pos) {
    Symbol *threadIndexSym = new Symbol("threadIndex", pos, AtomicType::UniformConstUInt32);
    m->symbolTable->AddVariable(threadIndexSym);

    Symbol *threadCountSym = new Symbol("threadCount", pos, AtomicType::UniformConstUInt32);
    m->symbolTable->AddVariable(threadCountSym);
}


/** Small utility routine to construct a string for error messages that
    suggests alternate tokens for possibly-misspelled ones... */
static std::string lGetAlternates(std::vector<std::string> &alternates) {
    std::string alts;
    if (alternates.size()) {
        alts += " Did you mean ";
        for (unsigned int i = 0; i < alternates.size(); ++i) {
            alts += std::string("\"") + alternates[i] + std::string("\"");
            if (i < alternates.size() - 1) alts += ", or ";
        }
        alts += "?";
    }
    return alts;
}

static const char *
lGetStorageClassString(StorageClass sc) {
    switch (sc) {
    case SC_NONE:
        return "";
    case SC_EXTERN:
        return "extern";
    case SC_EXPORT:
        return "export";
    case SC_STATIC:
        return "static";
    case SC_TYPEDEF:
        return "typedef";
    case SC_EXTERN_C:
        return "extern \"C\"";
    default:
        assert(!"logic error in lGetStorageClassString()");
        return "";
    }
}


