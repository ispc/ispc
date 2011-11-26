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
/* one for 'if', one for 'cif' */
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

static void lAddDeclaration(DeclSpecs *ds, Declarator *decl);
static void lAddFunctionParams(Declarator *decl);
static void lAddMaskToSymbolTable(SourcePos pos);
static void lAddThreadIndexCountToSymbolTable(SourcePos pos);
static std::string lGetAlternates(std::vector<std::string> &alternates);
static const char *lGetStorageClassString(StorageClass sc);
static bool lGetConstantInt(Expr *expr, int *value, SourcePos pos, const char *usage);
static EnumType *lCreateEnumType(const char *name, std::vector<Symbol *> *enums,
                                 SourcePos pos);
static void lFinalizeEnumeratorSymbols(std::vector<Symbol *> &enums,
                                       const EnumType *enumType);

static const char *lBuiltinTokens[] = {
    "bool", "break", "case", "cbreak", "ccontinue", "cdo", "cfor",
    "cif", "cwhile", "const", "continue", "creturn", "default", "do", "double", 
    "else", "enum", "export", "extern", "false", "float", "for", "goto", "if",
    "inline", "int", "int8", "int16", "int32", "int64", "launch", "NULL",
    "print", "return", "sizeof",
    "static", "struct", "switch", "sync", "task", "true", "typedef", "uniform",
    "unsigned", "varying", "void", "while", NULL 
};

static const char *lParamListTokens[] = {
    "bool", "const", "double", "enum", "false", "float", "int",
    "int8", "int16", "int32", "int64", "struct", "true",
    "uniform", "unsigned", "varying", "void", NULL 
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
    const EnumType *enumType;
    Symbol *enumerator;
    std::vector<Symbol *> *enumeratorList;
    int32_t int32Val;
    double floatVal;
    int64_t int64Val;
    std::string *stringVal;
    const char *constCharPtr;
}


%token TOKEN_INT32_CONSTANT TOKEN_UINT32_CONSTANT TOKEN_INT64_CONSTANT
%token TOKEN_UINT64_CONSTANT TOKEN_FLOAT_CONSTANT TOKEN_STRING_C_LITERAL
%token TOKEN_IDENTIFIER TOKEN_STRING_LITERAL TOKEN_TYPE_NAME TOKEN_NULL
%token TOKEN_PTR_OP TOKEN_INC_OP TOKEN_DEC_OP TOKEN_LEFT_OP TOKEN_RIGHT_OP 
%token TOKEN_LE_OP TOKEN_GE_OP TOKEN_EQ_OP TOKEN_NE_OP
%token TOKEN_AND_OP TOKEN_OR_OP TOKEN_MUL_ASSIGN TOKEN_DIV_ASSIGN TOKEN_MOD_ASSIGN 
%token TOKEN_ADD_ASSIGN TOKEN_SUB_ASSIGN TOKEN_LEFT_ASSIGN TOKEN_RIGHT_ASSIGN 
%token TOKEN_AND_ASSIGN TOKEN_OR_ASSIGN TOKEN_XOR_ASSIGN
%token TOKEN_SIZEOF

%token TOKEN_EXTERN TOKEN_EXPORT TOKEN_STATIC TOKEN_INLINE TOKEN_TASK 
%token TOKEN_UNIFORM TOKEN_VARYING TOKEN_TYPEDEF TOKEN_SOA
%token TOKEN_CHAR TOKEN_INT TOKEN_UNSIGNED TOKEN_FLOAT TOKEN_DOUBLE
%token TOKEN_INT8 TOKEN_INT16 TOKEN_INT64 TOKEN_CONST TOKEN_VOID TOKEN_BOOL 
%token TOKEN_ENUM TOKEN_STRUCT TOKEN_TRUE TOKEN_FALSE

%token TOKEN_CASE TOKEN_DEFAULT TOKEN_IF TOKEN_ELSE TOKEN_SWITCH
%token TOKEN_WHILE TOKEN_DO TOKEN_LAUNCH
%token TOKEN_FOR TOKEN_GOTO TOKEN_CONTINUE TOKEN_BREAK TOKEN_RETURN
%token TOKEN_CIF TOKEN_CDO TOKEN_CFOR TOKEN_CWHILE TOKEN_CBREAK
%token TOKEN_CCONTINUE TOKEN_CRETURN TOKEN_SYNC TOKEN_PRINT TOKEN_ASSERT

%type <expr> primary_expression postfix_expression
%type <expr> unary_expression cast_expression launch_expression
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
%type <stmt> assert_statement sync_statement

%type <declaration> declaration parameter_declaration
%type <declarators> init_declarator_list 
%type <declarationList> parameter_list parameter_type_list
%type <declarator> declarator pointer reference
%type <declarator> init_declarator direct_declarator struct_declarator
%type <declarator> abstract_declarator direct_abstract_declarator

%type <structDeclaratorList> struct_declarator_list
%type <structDeclaration> struct_declaration
%type <structDeclarationList> struct_declaration_list

%type <enumeratorList> enumerator_list
%type <enumerator> enumerator
%type <enumType> enum_specifier

%type <type> specifier_qualifier_list struct_or_union_specifier
%type <type> type_specifier type_name
%type <type> short_vec_specifier
%type <atomicType> atomic_var_type_specifier

%type <typeQualifier> type_qualifier type_qualifier_list
%type <storageClass> storage_class_specifier
%type <declSpecs> declaration_specifiers 

%type <stringVal> string_constant
%type <constCharPtr> struct_or_union_name enum_identifier
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
                $$ = new FunctionSymbolExpr(name, funs, @1);
        }
        if ($$ == NULL) {
            std::vector<std::string> alternates = 
                m->symbolTable->ClosestVariableOrFunctionMatch(name);
            std::string alts = lGetAlternates(alternates);
            Error(@1, "Undeclared symbol \"%s\".%s", name, alts.c_str());
        }
    }
    | TOKEN_INT32_CONSTANT {
        $$ = new ConstExpr(AtomicType::UniformConstInt32, yylval.int32Val, @1); 
    }
    | TOKEN_UINT32_CONSTANT {
        $$ = new ConstExpr(AtomicType::UniformConstUInt32, (uint32_t)yylval.int32Val, @1); 
    }
    | TOKEN_INT64_CONSTANT {
        $$ = new ConstExpr(AtomicType::UniformConstInt64, yylval.int64Val, @1); 
    }
    | TOKEN_UINT64_CONSTANT {
        $$ = new ConstExpr(AtomicType::UniformConstUInt64, (uint64_t)yylval.int64Val, @1); 
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
    | TOKEN_NULL {
        $$ = new NullPointerExpr(@1);
    }
/*    | TOKEN_STRING_LITERAL
       { UNIMPLEMENTED }*/
    | '(' expression ')' { $$ = $2; }
    ;

launch_expression
    : TOKEN_LAUNCH '<' postfix_expression '(' argument_expression_list ')' '>'
      { 
          ConstExpr *oneExpr = new ConstExpr(AtomicType::UniformInt32, (int32_t)1, @3);
          $$ = new FunctionCallExpr($3, $5, Union(@3, @6), true, oneExpr);
      }
    | TOKEN_LAUNCH '<' postfix_expression '(' ')' '>'
      {
          ConstExpr *oneExpr = new ConstExpr(AtomicType::UniformInt32, (int32_t)1, @3);
          $$ = new FunctionCallExpr($3, new ExprList(Union(@4,@5)), Union(@3, @5), true, oneExpr);
       }
    | TOKEN_LAUNCH '[' expression ']' '<' postfix_expression '(' argument_expression_list ')' '>'
      { $$ = new FunctionCallExpr($6, $8, Union(@6,@9), true, $3); }
    | TOKEN_LAUNCH '[' expression ']' '<' postfix_expression '(' ')' '>'
      { $$ = new FunctionCallExpr($6, new ExprList(Union(@6,@7)), Union(@6,@8), true, $3); }
    ;

postfix_expression
    : primary_expression
    | postfix_expression '[' expression ']'
      { $$ = new IndexExpr($1, $3, Union(@1,@4)); }
    | postfix_expression '(' ')'
      { $$ = new FunctionCallExpr($1, new ExprList(Union(@1,@2)), Union(@1,@3)); }
    | postfix_expression '(' argument_expression_list ')'
      { $$ = new FunctionCallExpr($1, $3, Union(@1,@4)); }
    | launch_expression
    | postfix_expression '.' TOKEN_IDENTIFIER
      { $$ = MemberExpr::create($1, yytext, Union(@1,@3), @3, false); }
    | postfix_expression TOKEN_PTR_OP TOKEN_IDENTIFIER
      { $$ = MemberExpr::create($1, yytext, Union(@1,@3), @3, true); }
    | postfix_expression TOKEN_INC_OP
      { $$ = new UnaryExpr(UnaryExpr::PostInc, $1, Union(@1,@2)); }
    | postfix_expression TOKEN_DEC_OP
      { $$ = new UnaryExpr(UnaryExpr::PostDec, $1, Union(@1,@2)); }
    ;

argument_expression_list
    : assignment_expression      { $$ = new ExprList($1, @1); }
    | argument_expression_list ',' assignment_expression
      {
          ExprList *argList = dynamic_cast<ExprList *>($1);
          assert(argList != NULL);
          argList->exprs.push_back($3);
          argList->pos = Union(argList->pos, @3);
          $$ = argList;
      }
    ;

unary_expression
    : postfix_expression
    | TOKEN_INC_OP unary_expression   
      { $$ = new UnaryExpr(UnaryExpr::PreInc, $2, Union(@1, @2)); }
    | TOKEN_DEC_OP unary_expression   
      { $$ = new UnaryExpr(UnaryExpr::PreDec, $2, Union(@1, @2)); }
    | '&' unary_expression
      { $$ = new AddressOfExpr($2, Union(@1, @2)); }
    | '*' unary_expression
      { $$ = new DereferenceExpr($2, Union(@1, @2)); }
    | '+' cast_expression 
      { $$ = $2; }
    | '-' cast_expression 
      { $$ = new UnaryExpr(UnaryExpr::Negate, $2, Union(@1, @2)); }
    | '~' cast_expression 
      { $$ = new UnaryExpr(UnaryExpr::BitNot, $2, Union(@1, @2)); }
    | '!' cast_expression 
      { $$ = new UnaryExpr(UnaryExpr::LogicalNot, $2, Union(@1, @2)); }
    | TOKEN_SIZEOF unary_expression
      { $$ = new SizeOfExpr($2, Union(@1, @2)); }
    | TOKEN_SIZEOF '(' type_name ')'
      { $$ = new SizeOfExpr($3, Union(@1, @4)); }
    ;

cast_expression
    : unary_expression
    | '(' type_name ')' cast_expression
      {
          // Pass true here to try to preserve uniformity 
          // so that things like:
          // uniform int y = ...;
          // uniform float x = 1. / (float)y;
          // don't issue an error due to (float)y being inadvertently
          // and undesirably-to-the-user "varying"...
          $$ = new TypeCastExpr($2, $4, true, Union(@1,@4)); 
      }
    ;

multiplicative_expression
    : cast_expression
    | multiplicative_expression '*' cast_expression
      { $$ = new BinaryExpr(BinaryExpr::Mul, $1, $3, Union(@1, @3)); }
    | multiplicative_expression '/' cast_expression
      { $$ = new BinaryExpr(BinaryExpr::Div, $1, $3, Union(@1, @3)); }
    | multiplicative_expression '%' cast_expression
      { $$ = new BinaryExpr(BinaryExpr::Mod, $1, $3, Union(@1, @3)); }
    ;

additive_expression
    : multiplicative_expression
    | additive_expression '+' multiplicative_expression
      { $$ = new BinaryExpr(BinaryExpr::Add, $1, $3, Union(@1, @3)); }
    | additive_expression '-' multiplicative_expression
      { $$ = new BinaryExpr(BinaryExpr::Sub, $1, $3, Union(@1, @3)); }
    ;

shift_expression
    : additive_expression
    | shift_expression TOKEN_LEFT_OP additive_expression
      { $$ = new BinaryExpr(BinaryExpr::Shl, $1, $3, Union(@1,@3)); }
    | shift_expression TOKEN_RIGHT_OP additive_expression
      { $$ = new BinaryExpr(BinaryExpr::Shr, $1, $3, Union(@1,@3)); }
    ;

relational_expression
    : shift_expression
    | relational_expression '<' shift_expression
      { $$ = new BinaryExpr(BinaryExpr::Lt, $1, $3, Union(@1, @3)); }
    | relational_expression '>' shift_expression
      { $$ = new BinaryExpr(BinaryExpr::Gt, $1, $3, Union(@1, @3)); }
    | relational_expression TOKEN_LE_OP shift_expression
      { $$ = new BinaryExpr(BinaryExpr::Le, $1, $3, Union(@1, @3)); }
    | relational_expression TOKEN_GE_OP shift_expression
      { $$ = new BinaryExpr(BinaryExpr::Ge, $1, $3, Union(@1, @3)); }
    ;

equality_expression
    : relational_expression
    | equality_expression TOKEN_EQ_OP relational_expression
      { $$ = new BinaryExpr(BinaryExpr::Equal, $1, $3, Union(@1,@3)); }
    | equality_expression TOKEN_NE_OP relational_expression
      { $$ = new BinaryExpr(BinaryExpr::NotEqual, $1, $3, Union(@1,@3)); }
    ;

and_expression
    : equality_expression
    | and_expression '&' equality_expression
      { $$ = new BinaryExpr(BinaryExpr::BitAnd, $1, $3, Union(@1, @3)); }
    ;

exclusive_or_expression
    : and_expression
    | exclusive_or_expression '^' and_expression
      { $$ = new BinaryExpr(BinaryExpr::BitXor, $1, $3, Union(@1, @3)); }
    ;

inclusive_or_expression
    : exclusive_or_expression
    | inclusive_or_expression '|' exclusive_or_expression
      { $$ = new BinaryExpr(BinaryExpr::BitOr, $1, $3, Union(@1, @3)); }
    ;

logical_and_expression
    : inclusive_or_expression
    | logical_and_expression TOKEN_AND_OP inclusive_or_expression
      { $$ = new BinaryExpr(BinaryExpr::LogicalAnd, $1, $3, Union(@1, @3)); }
    ;

logical_or_expression
    : logical_and_expression
    | logical_or_expression TOKEN_OR_OP logical_and_expression
      { $$ = new BinaryExpr(BinaryExpr::LogicalOr, $1, $3, Union(@1, @3)); }
    ;

conditional_expression
    : logical_or_expression
    | logical_or_expression '?' expression ':' conditional_expression
      { $$ = new SelectExpr($1, $3, $5, Union(@1,@5)); }
    ;

assignment_expression
    : conditional_expression
    | unary_expression '=' assignment_expression
      { $$ = new AssignExpr(AssignExpr::Assign, $1, $3, Union(@1, @3)); }
    | unary_expression TOKEN_MUL_ASSIGN assignment_expression
      { $$ = new AssignExpr(AssignExpr::MulAssign, $1, $3, Union(@1, @3)); }
    | unary_expression TOKEN_DIV_ASSIGN assignment_expression
      { $$ = new AssignExpr(AssignExpr::DivAssign, $1, $3, Union(@1, @3)); }
    | unary_expression TOKEN_MOD_ASSIGN assignment_expression
      { $$ = new AssignExpr(AssignExpr::ModAssign, $1, $3, Union(@1, @3)); }
    | unary_expression TOKEN_ADD_ASSIGN assignment_expression
      { $$ = new AssignExpr(AssignExpr::AddAssign, $1, $3, Union(@1, @3)); }
    | unary_expression TOKEN_SUB_ASSIGN assignment_expression
      { $$ = new AssignExpr(AssignExpr::SubAssign, $1, $3, Union(@1, @3)); }
    | unary_expression TOKEN_LEFT_ASSIGN assignment_expression
      { $$ = new AssignExpr(AssignExpr::ShlAssign, $1, $3, Union(@1, @3)); }
    | unary_expression TOKEN_RIGHT_ASSIGN assignment_expression
      { $$ = new AssignExpr(AssignExpr::ShrAssign, $1, $3, Union(@1, @3)); }
    | unary_expression TOKEN_AND_ASSIGN assignment_expression
      { $$ = new AssignExpr(AssignExpr::AndAssign, $1, $3, Union(@1, @3)); }
    | unary_expression TOKEN_XOR_ASSIGN assignment_expression
      { $$ = new AssignExpr(AssignExpr::XorAssign, $1, $3, Union(@1, @3)); }
    | unary_expression TOKEN_OR_ASSIGN assignment_expression
      { $$ = new AssignExpr(AssignExpr::OrAssign, $1, $3, Union(@1, @3)); }
    ;

expression
    : assignment_expression
    | expression ',' assignment_expression
      { $$ = new BinaryExpr(BinaryExpr::Comma, $1, $3, Union(@1, @3)); }
    ;

constant_expression
    : conditional_expression
    ;

declaration_statement
    : declaration     
    {
        if ($1->declSpecs->storageClass == SC_TYPEDEF) {
            for (unsigned int i = 0; i < $1->declarators.size(); ++i) {
                m->AddTypeDef($1->declarators[i]->sym);
            }
        }
        else {
            std::vector<VariableDeclaration> vars = $1->GetVariableDeclarations();
            $$ = new DeclStmt(vars, @1);
        }
    }
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
          if (ds != NULL) {
              if (ds->storageClass != SC_NONE)
                  Error(@1, "Multiple storage class specifiers in a declaration are illegal. "
                        "(Have provided both \"%s\" and \"%s\".)",
                        lGetStorageClassString(ds->storageClass),
                        lGetStorageClassString($1));
              else
                  ds->storageClass = $1;
          }
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
          if (ds != NULL) {
              if (ds->soaWidth != 0)
                  Error(@1, "soa<> qualifier supplied multiple times in declaration.");
              else
                  ds->soaWidth = $1;
          }
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
          if (ds != NULL) {
              if (ds->baseType != NULL)
                  Error(@1, "Multiple types provided for declaration.");
              ds->baseType = $1;
          }
          $$ = ds;
      }
    | type_qualifier
      {
          $$ = new DeclSpecs(NULL, SC_NONE, $1);
      }
    | type_qualifier declaration_specifiers
      {
          DeclSpecs *ds = (DeclSpecs *)$2;
          if (ds != NULL)
              ds->typeQualifiers |= $1;
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
          if (dl != NULL && $3 != NULL)
              dl->push_back($3);
          $$ = $1;
      }
    ;

init_declarator
    : declarator
    | declarator '=' initializer 
      {
          if ($1 != NULL)
              $1->initExpr = $3; 
          $$ = $1; 
      }
    ;

storage_class_specifier
    : TOKEN_TYPEDEF { $$ = SC_TYPEDEF; }
    | TOKEN_EXTERN { $$ = SC_EXTERN; }
    | TOKEN_EXTERN TOKEN_STRING_C_LITERAL  { $$ = SC_EXTERN_C; }
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
    | enum_specifier { $$ = $1; }
    ;

atomic_var_type_specifier
    : TOKEN_VOID { $$ = AtomicType::Void; }
    | TOKEN_BOOL { $$ = AtomicType::VaryingBool; }
    | TOKEN_INT8 { $$ = AtomicType::VaryingInt8; }
    | TOKEN_INT16 { $$ = AtomicType::VaryingInt16; }
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
            Error(@2, "Type \"%s\" is not a struct type! (%s)", $2,
                  st->GetString().c_str());
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
          if (sdl != NULL && $1 != NULL)
              sdl->push_back($1);
          $$ = sdl;
      }
    | struct_declaration_list struct_declaration 
      {
          std::vector<StructDeclaration *> *sdl = (std::vector<StructDeclaration *> *)$1;
          if (sdl != NULL && $2 != NULL)
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
        if ($2 != NULL) {
            if ($1 == TYPEQUAL_UNIFORM)
                $$ = $2->GetAsUniformType();
            else if ($1 == TYPEQUAL_VARYING)
                $$ = $2->GetAsVaryingType();
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
            else if ($1 == TYPEQUAL_INLINE) {
                Error(@1, "\"inline\" qualifier is illegal outside of "
                      "function declarations.");
                $$ = $2;
            }
            else if ($1 == TYPEQUAL_TASK) {
                Error(@1, "\"task\" qualifier is illegal outside of "
                      "function declarations.");
                $$ = $2;
            }
            else
                FATAL("Unhandled type qualifier in parser.");
        }
        else
            $$ = NULL;
    }
    ;


struct_declarator_list
    : struct_declarator 
      {
          std::vector<Declarator *> *sdl = new std::vector<Declarator *>;
          if ($1 != NULL)
              sdl->push_back($1);
          $$ = sdl;
      }
    | struct_declarator_list ',' struct_declarator 
      {
          std::vector<Declarator *> *sdl = (std::vector<Declarator *> *)$1;
          if (sdl != NULL && $3 != NULL)
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

enum_identifier
    : TOKEN_IDENTIFIER { $$ = strdup(yytext); }

enum_specifier
    : TOKEN_ENUM '{' enumerator_list '}' 
      {
          $$ = lCreateEnumType(NULL, $3, @1);
      }
    | TOKEN_ENUM enum_identifier '{' enumerator_list '}' 
      {
          $$ = lCreateEnumType($2, $4, @2);
      }
    | TOKEN_ENUM '{' enumerator_list ',' '}' 
      {
          $$ = lCreateEnumType(NULL, $3, @1);
      }
    | TOKEN_ENUM enum_identifier '{' enumerator_list ',' '}' 
      {
          $$ = lCreateEnumType($2, $4, @2);
      }
    | TOKEN_ENUM enum_identifier
      {
          const Type *type = m->symbolTable->LookupType($2);
          if (type == NULL) {
              std::vector<std::string> alternates = m->symbolTable->ClosestEnumTypeMatch($2);
              std::string alts = lGetAlternates(alternates);
              Error(@2, "Enum type \"%s\" unknown.%s", $2, alts.c_str());
              $$ = NULL;
          }
          else {
              const EnumType *enumType = dynamic_cast<const EnumType *>(type);
              if (enumType == NULL) {
                  Error(@2, "Type \"%s\" is not an enum type (%s).", $2,
                        type->GetString().c_str());
                  $$ = NULL;
              }
              else
                  $$ = enumType;
          }
      }
    ;

enumerator_list
    : enumerator 
      {
          if ($1 == NULL)
              $$ = NULL;
          else {
              std::vector<Symbol *> *el = new std::vector<Symbol *>; 
              el->push_back($1);
              $$ = el;
          }
      }
    | enumerator_list ',' enumerator
      {
          if ($1 != NULL && $3 != NULL)
              $1->push_back($3);
          $$ = $1;
      }
    ;

enumerator
    : enum_identifier
      {
          $$ = new Symbol($1, @1);
      }
    | enum_identifier '=' constant_expression
      {
          int value;
          if ($1 != NULL && $3 != NULL &&
              lGetConstantInt($3, &value, @3, "Enumerator value")) {
              Symbol *sym = new Symbol($1, @1);
              sym->constValue = new ConstExpr(AtomicType::UniformConstUInt32,
                                              (uint32_t)value, @3);
              $$ = sym;
          }
          else
              $$ = NULL;
      }
    ;

type_qualifier
    : TOKEN_CONST      { $$ = TYPEQUAL_CONST; }
    | TOKEN_UNIFORM    { $$ = TYPEQUAL_UNIFORM; }
    | TOKEN_VARYING    { $$ = TYPEQUAL_VARYING; }
    | TOKEN_TASK       { $$ = TYPEQUAL_TASK; }
    | TOKEN_INLINE     { $$ = TYPEQUAL_INLINE; }
    | TOKEN_UNSIGNED   { $$ = TYPEQUAL_UNSIGNED; }
    ;

type_qualifier_list
    : type_qualifier
    {
        $$ = $1;
    }
    | type_qualifier_list type_qualifier 
    {
        $$ = $1 | $2;
    }
    ;

declarator
    : pointer direct_declarator
    {
        Declarator *tail = $1;
        while (tail->child != NULL)
           tail = tail->child;
        tail->child = $2;
        $$ = $1;
    }
    | reference direct_declarator
    {
        Declarator *tail = $1;
        while (tail->child != NULL)
           tail = tail->child;
        tail->child = $2;
        $$ = $1;
    }
    | direct_declarator
    ;

int_constant
    : TOKEN_INT32_CONSTANT { $$ = yylval.int32Val; }
    ;

direct_declarator
    : TOKEN_IDENTIFIER
      {
          Declarator *d = new Declarator(DK_BASE, @1);
          d->sym = new Symbol(yytext, @1);
          $$ = d;
      }
    | '(' declarator ')' 
    {
        $$ = $2; 
    }
    | direct_declarator '[' constant_expression ']'
    {
        int size;
        if ($1 != NULL && lGetConstantInt($3, &size, @3, "Array dimension")) {
            Declarator *d = new Declarator(DK_ARRAY, Union(@1, @4));
            d->arraySize = size;
            d->child = $1;
            $$ = d;
        }
        else
            $$ = NULL;
    }
    | direct_declarator '[' ']'
    {
        if ($1 != NULL) {
            Declarator *d = new Declarator(DK_ARRAY, Union(@1, @3));
            d->arraySize = 0; // unsize
            d->child = $1;
            $$ = d;
        }
        else
            $$ = NULL;
    }
    | direct_declarator '(' parameter_type_list ')'
      {
          if ($1 != NULL) {
              Declarator *d = new Declarator(DK_FUNCTION, Union(@1, @4));
              d->child = $1;
              if ($3 != NULL) d->functionParams = *$3;
              $$ = d;
          }
          else
              $$ = NULL;
      }
    | direct_declarator '(' ')'
      {
          if ($1 != NULL) {
              Declarator *d = new Declarator(DK_FUNCTION, Union(@1, @3));
              d->child = $1;
              $$ = d;
          }
          else
              $$ = NULL;
      }
    ;


pointer
    : '*' 
    {
        $$ = new Declarator(DK_POINTER, @1); 
    }
    | '*' type_qualifier_list
      {
          Declarator *d = new Declarator(DK_POINTER, Union(@1, @2));
          d->typeQualifiers = $2;
          $$ = d;
      }
    | '*' pointer
      {
          Declarator *d = new Declarator(DK_POINTER, Union(@1, @2));
          d->child = $2;
          $$ = d;
      }
    | '*' type_qualifier_list pointer
      {
          Declarator *d = new Declarator(DK_POINTER, Union(@1, @3));
          d->typeQualifiers = $2;
          d->child = $3;
          $$ = d;
      }
    ;


reference
    : '&' 
    {
        $$ = new Declarator(DK_REFERENCE, @1); 
    }
    ;


parameter_type_list
    : parameter_list { $$ = $1; }
    ;

parameter_list
    : parameter_declaration
    {
        std::vector<Declaration *> *dl = new std::vector<Declaration *>;
        if ($1 != NULL)
            dl->push_back($1);
        $$ = dl;
    }
    | parameter_list ',' parameter_declaration
    {
        std::vector<Declaration *> *dl = (std::vector<Declaration *> *)$1;
        if (dl == NULL)
            // dl may be NULL due to an earlier parse error...
            dl = new std::vector<Declaration *>;
        if ($3 != NULL)
            dl->push_back($3);
        $$ = dl;
    }
    | error
    {
        std::vector<std::string> builtinTokens;
        const char **token = lParamListTokens;
        while (*token) {
            builtinTokens.push_back(*token);
            ++token;
        }
        if (strlen(yytext) == 0)
            Error(@1, "Syntax error--premature end of file.");
        else {
            std::vector<std::string> alternates = MatchStrings(yytext, builtinTokens);
            std::string alts = lGetAlternates(alternates);
            Error(@1, "Syntax error--token \"%s\" unexpected.%s", yytext, alts.c_str());
        }
        $$ = NULL;
    }
    ;

parameter_declaration
    : declaration_specifiers declarator
    {
        $$ = new Declaration($1, $2); 
    }
    | declaration_specifiers declarator '=' initializer
    { 
        if ($2 != NULL)
            $2->initExpr = $4;
        $$ = new Declaration($1, $2); 

    }
    | declaration_specifiers abstract_declarator
    {
        $$ = new Declaration($1, $2);
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
    {
        $$ = $2->GetType($1, NULL);
    }
    ;

abstract_declarator
    : pointer
      {
          Declarator *d = new Declarator(DK_POINTER, @1);
          $$ = d;
      }
    | direct_abstract_declarator
    | pointer direct_abstract_declarator
      {
          Declarator *d = new Declarator(DK_POINTER, Union(@1, @2));
          d->child = $2;
          $$ = d;
      }
    | reference
      {
          Declarator *d = new Declarator(DK_REFERENCE, @1);
          $$ = d;
      }
    | reference direct_abstract_declarator
      {
          Declarator *d = new Declarator(DK_REFERENCE, Union(@1, @2));
          d->child = $2;
          $$ = d;
      }
    ;

direct_abstract_declarator
    : '(' abstract_declarator ')'
      { $$ = $2; }
    | '[' ']'
      {
          Declarator *d = new Declarator(DK_ARRAY, Union(@1, @2));
          d->arraySize = 0;
          $$ = d;
      }
    | '[' constant_expression ']'
      {
        int size;
        if (lGetConstantInt($2, &size, @2, "Array dimension")) {
            Declarator *d = new Declarator(DK_ARRAY, Union(@1, @3));
            d->arraySize = size;
            $$ = d;
        }
        else
            $$ = NULL;
      }
    | direct_abstract_declarator '[' ']'
      {
          Declarator *d = new Declarator(DK_ARRAY, Union(@1, @3));
          d->arraySize = 0;
          d->child = $1;
          $$ = d;
      }
    | direct_abstract_declarator '[' constant_expression ']'
      {
          int size;
          if (lGetConstantInt($3, &size, @3, "Array dimension")) {
              Declarator *d = new Declarator(DK_ARRAY, Union(@1, @4));
              d->arraySize = size;
              d->child = $1;
              $$ = d;
          }
          else
              $$ = NULL;
      }
    | '(' ')'
      { $$ = new Declarator(DK_FUNCTION, Union(@1, @2)); }
    | '(' parameter_type_list ')'
      {
          Declarator *d = new Declarator(DK_FUNCTION, Union(@1, @3));
          if ($2 != NULL) d->functionParams = *$2;
      }
    | direct_abstract_declarator '(' ')'
      {
          Declarator *d = new Declarator(DK_FUNCTION, Union(@1, @3));
          d->child = $1;
          $$ = d;
      }
    | direct_abstract_declarator '(' parameter_type_list ')'
      {
          Declarator *d = new Declarator(DK_FUNCTION, Union(@1, @4));
          d->child = $1;
          if ($3 != NULL) d->functionParams = *$3;
          $$ = d;
      }
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
          if ($1 == NULL)
              $$ = NULL;
          else {
              ExprList *exprList = dynamic_cast<ExprList *>($1);
              assert(exprList);
              exprList->exprs.push_back($3);
              exprList->pos = Union(exprList->pos, @3);
              $$ = exprList;
          }
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
    | assert_statement
    | sync_statement
    | error
    {
        std::vector<std::string> builtinTokens;
        const char **token = lBuiltinTokens;
        while (*token) {
            builtinTokens.push_back(*token);
            ++token;
        }
        if (strlen(yytext) == 0)
            Error(@1, "Syntax error--premature end of file.");
        else {
            std::vector<std::string> alternates = MatchStrings(yytext, builtinTokens);
            std::string alts = lGetAlternates(alternates);
            Error(@1, "Syntax error--token \"%s\" unexpected.%s", yytext, alts.c_str());
        }
        $$ = NULL;
    }
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
          if ($1 != NULL)
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

sync_statement
    : TOKEN_SYNC 
      { $$ = new ExprStmt(new SyncExpr(@1), @1); }
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

assert_statement
    : TOKEN_ASSERT '(' string_constant ',' expression ')'
      {
          $$ = new AssertStmt(*$3, $5, @1);
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
        if (strlen(yytext) == 0)
            Error(@1, "Syntax error--premature end of file.");
        else {
            std::vector<std::string> alternates = MatchStrings(yytext, builtinTokens);
            std::string alts = lGetAlternates(alternates);
            Error(@1, "Syntax error--token \"%s\" unexpected.%s", yytext, alts.c_str());
        }
    }
    ;

external_declaration
    : function_definition
    | TOKEN_EXTERN TOKEN_STRING_C_LITERAL '{' declaration '}'
    | declaration 
    { 
        if ($1 != NULL)
            for (unsigned int i = 0; i < $1->declarators.size(); ++i)
                lAddDeclaration($1->declSpecs, $1->declarators[i]);
    }
    ;

function_definition
    : declaration_specifiers declarator 
    {
        lAddDeclaration($1, $2);
        lAddFunctionParams($2); 
        lAddMaskToSymbolTable(@2);
        if ($1->typeQualifiers & TYPEQUAL_TASK)
            lAddThreadIndexCountToSymbolTable(@2);
    } 
    compound_statement
    {
        std::vector<Symbol *> args;
        Symbol *sym = $2->GetFunctionInfo($1, &args);
        if (sym != NULL)
            m->AddFunctionDefinition(sym, args, $4);
        m->symbolTable->PopScope(); // push in lAddFunctionParams();
    }
/* function with no declared return type??
func(...) 
    | declarator { lAddFunctionParams($1); } compound_statement
    {
        m->AddFunction(new DeclSpecs(XXX, $1, $3);
        m->symbolTable->PopScope(); // push in lAddFunctionParams();
    }
*/
    ;

%%


static void
lAddDeclaration(DeclSpecs *ds, Declarator *decl) {
    if (ds == NULL || decl == NULL)
        // Error happened earlier during parsing
        return;

    if (ds->storageClass == SC_TYPEDEF)
        m->AddTypeDef(decl->GetSymbol());
    else {
        const Type *t = decl->GetType(ds);
        if (t == NULL)
            return;
        const FunctionType *ft = dynamic_cast<const FunctionType *>(t);
        if (ft != NULL) {
            Symbol *funSym = decl->GetSymbol();
            assert(funSym != NULL);
            funSym->type = ft;
            funSym->storageClass = ds->storageClass;

            bool isInline = (ds->typeQualifiers & TYPEQUAL_INLINE);
            m->AddFunctionDeclaration(funSym, isInline);
        }
        else
            m->AddGlobalVariable(decl->GetSymbol(), decl->initExpr,
                                 (ds->typeQualifiers & TYPEQUAL_CONST) != 0);
    }
}


/** We're about to start parsing the body of a function; add all of the
    parameters to the symbol table so that they're available.
*/
static void
lAddFunctionParams(Declarator *decl) {
    m->symbolTable->PushScope();

    // walk down to the declarator for the function itself 
    while (decl->kind != DK_FUNCTION && decl->child != NULL)
        decl = decl->child;
    assert(decl->kind == DK_FUNCTION);

    // now loop over its parameters and add them to the symbol table
    for (unsigned int i = 0; i < decl->functionParams.size(); ++i) {
        Declaration *pdecl = decl->functionParams[i];
        if (pdecl == NULL)
            continue;
        assert(pdecl->declarators.size() == 1);
        Symbol *sym = pdecl->declarators[0]->GetSymbol();
#ifndef NDEBUG
        bool ok = m->symbolTable->AddVariable(sym);
        if (ok == false)
            assert(m->errorCount > 0);
#else
        m->symbolTable->AddVariable(sym);
#endif
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

    Symbol *taskIndexSym = new Symbol("taskIndex", pos, AtomicType::UniformConstUInt32);
    m->symbolTable->AddVariable(taskIndexSym);

    Symbol *taskCountSym = new Symbol("taskCount", pos, AtomicType::UniformConstUInt32);
    m->symbolTable->AddVariable(taskCountSym);
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


/** Given an expression, see if it is equal to a compile-time constant
    integer value.  If so, return true and return the value in *value.
    If the expression isn't a compile-time constant or isn't an integer
    type, return false.
*/
static bool
lGetConstantInt(Expr *expr, int *value, SourcePos pos, const char *usage) {
    if (expr == NULL)
        return false;
    expr = expr->TypeCheck();
    if (expr == NULL)
        return false;
    expr = expr->Optimize();
    if (expr == NULL)
        return false;

    llvm::Constant *cval = expr->GetConstant(expr->GetType());
    if (cval == NULL) {
        Error(pos, "%s must be a compile-time constant.", usage);
        return false;
    }
    else {
        llvm::ConstantInt *ci = llvm::dyn_cast<llvm::ConstantInt>(cval);
        if (ci == NULL) {
            Error(pos, "%s must be a compile-time integer constant.", usage);
            return false;
        }
        *value = (int)ci->getZExtValue();
        return true;
    }
}


static EnumType *
lCreateEnumType(const char *name, std::vector<Symbol *> *enums, SourcePos pos) {
    if (enums == NULL)
        return NULL;

    EnumType *enumType = name ? new EnumType(name, pos) : new EnumType(pos);
    if (name != NULL)
        m->symbolTable->AddType(name, enumType, pos);

    lFinalizeEnumeratorSymbols(*enums, enumType);
    for (unsigned int i = 0; i < enums->size(); ++i)
        m->symbolTable->AddVariable((*enums)[i]);
    enumType->SetEnumerators(*enums);
    return enumType;
}


/** Given an array of enumerator symbols, make sure each of them has a
    ConstExpr * in their Symbol::constValue member that stores their 
    unsigned integer value.  Symbols that had values explicitly provided
    in the source file will already have ConstExpr * set; we just need
    to set the values for the others here.
*/
static void
lFinalizeEnumeratorSymbols(std::vector<Symbol *> &enums, 
                           const EnumType *enumType) {
    enumType = enumType->GetAsConstType();
    enumType = enumType->GetAsUniformType();

    /* nextVal tracks the value for the next enumerant.  It starts from
       zero and goes up with each successive enumerant.  If any of them
       has a value specified, then nextVal is ignored for that one and is
       set to one plus that one's value for the default value for the next
       one. */
    uint32_t nextVal = 0;

    for (unsigned int i = 0; i < enums.size(); ++i) {
        enums[i]->type = enumType;
        if (enums[i]->constValue != NULL) {
            /* Already has a value, so first update nextVal with it. */
            int count = enums[i]->constValue->AsUInt32(&nextVal);
            assert(count == 1);
            ++nextVal;

            /* When the source file as being parsed, the ConstExpr for any
               enumerant with a specified value was set to have unsigned
               int32 type, since we haven't created the parent EnumType
               by then.  Therefore, add a little type cast from uint32 to
               the actual enum type here and optimize it, which will have
               us end up with a ConstExpr with the desired EnumType... */
            Expr *castExpr = new TypeCastExpr(enumType, enums[i]->constValue,
                                              false, enums[i]->pos);
            castExpr = castExpr->Optimize();
            enums[i]->constValue = dynamic_cast<ConstExpr *>(castExpr);
            assert(enums[i]->constValue != NULL);
        }
        else {
            enums[i]->constValue = new ConstExpr(enumType, nextVal++, 
                                                 enums[i]->pos);
        }
    }
}
