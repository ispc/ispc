/*
  Copyright (c) 2010-2024, Intel Corporation

  SPDX-License-Identifier: BSD-3-Clause
*/

%locations

/* supress shift-reduces conflict message for dangling else */
/* one for 'if', one for 'cif' */
%expect 2

%define parse.error verbose

%code requires {

#define yytnamerr lYYTNameErr


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
          (Current).name = nullptr;                        /* new */ \
        }                                                              \
    while (0)

struct ForeachDimension;

struct PragmaAttributes {
    enum class AttributeType { none, pragmaloop, pragmawarning };
    PragmaAttributes() {
        aType = AttributeType::none;
        unrollType =  Globals::pragmaUnrollType::none;
        count = -1;
    }
    AttributeType aType;
    Globals::pragmaUnrollType unrollType;
    int count;
};

typedef std::pair<Declarator *, TemplateArgs *> SimpleTemplateIDType;

}


%{

#include "decl.h"
#include "expr.h"
#include "func.h"
#include "ispc.h"
#include "module.h"
#include "stmt.h"
#include "sym.h"
#include "type.h"
#include "util.h"

#include <stdio.h>
#include <llvm/IR/Constants.h>

using namespace ispc;

#define UNIMPLEMENTED \
        Error(yylloc, "Unimplemented parser functionality %s:%d", \
        __FILE__, __LINE__);

union YYSTYPE;
extern int yylex();

extern char *yytext;

void yyerror(const char *s);

void lCleanUpString(std::string *str);
void lFreeSimpleTemplateID(void *p);
static int lYYTNameErr(char *yyres, const char *yystr);

static void lSuggestBuiltinAlternates();
static void lSuggestParamListAlternates();

static void lAddDeclaration(DeclSpecs *ds, Declarator *decl);
static void lAddTemplateDeclaration(TemplateParms *templateParmList, DeclSpecs *ds, Declarator *decl);
static void lAddTemplateSpecialization(const TemplateArgs &templArgs, DeclSpecs *ds, Declarator *decl);
static void lAddFunctionParams(Declarator *decl);
static void lAddMaskToSymbolTable(SourcePos pos);
static void lAddThreadIndexCountToSymbolTable(SourcePos pos);
static std::string lGetAlternates(std::vector<std::string> &alternates);
static const char *lGetStorageClassString(StorageClass sc);
static bool lGetConstantInt(Expr *expr, int *value, SourcePos pos, const char *usage);

enum class TemplateType { Template, Instantiation, Specialization };
static void lCheckTemplateDeclSpecs(DeclSpecs *ds, SourcePos pos, TemplateType type, const char* name);

static EnumType *lCreateEnumType(const char *name, std::vector<Symbol *> *enums,
                                 SourcePos pos);
static void lFinalizeEnumeratorSymbols(std::vector<Symbol *> &enums,
                                       const EnumType *enumType);

static const char *lBuiltinTokens[] = {
    "assert", "bool", "break", "case", "cdo",
    "cfor", "cif", "cwhile", "const", "continue", "default",
    "do", "delete", "double", "else", "enum", "export", "extern", "false",
    "float16", "float", "for", "foreach", "foreach_active", "foreach_tiled",
    "foreach_unique", "goto", "if", "in", "inline",
    "int", "int8", "int16", "int32", "int64", "invoke_sycl", "launch", "new", "NULL",
    "print", "return", "signed", "sizeof", "static", "struct", "switch",
    "sync", "task", "true", "typedef", "uniform", "unmasked", "unsigned",
    "varying", "void", "while", NULL
};

static const char *lParamListTokens[] = {
    "bool", "const", "double", "enum", "false", "float16", "float", "int",
    "int8", "int16", "int32", "int64", "signed", "struct", "true",
    "uniform", "unsigned", "varying", "void", NULL
};

struct ForeachDimension {
    ForeachDimension(Symbol *s = nullptr, Expr *b = nullptr, Expr *e = nullptr) {
        sym = s;
        beginExpr = b;
        endExpr = e;
    }
    Symbol *sym;
    Expr *beginExpr, *endExpr;
};

%}

%union {
    uint64_t intVal;
    float  floatVal;
    double doubleVal;
    std::string *stringVal;
    const char *constCharPtr;

    Expr *expr;
    ExprList *exprList;
    const Type *type;
    std::vector<std::pair<const Type *, SourcePos> > *typeList;
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
    Symbol *symbol;
    std::vector<Symbol *> *symbolList;
    ForeachDimension *foreachDimension;
    std::vector<ForeachDimension *> *foreachDimensionList;
    std::pair<std::string, SourcePos> *declspecPair;
    std::vector<std::pair<std::string, SourcePos> > *declspecList;
    PragmaAttributes *pragmaAttributes;
    const TemplateArg *templateArg;
    const TemplateArgs *templateArgs;
    const TemplateParam *templateParm;
    TemplateParms *templateParmList;
    const TemplateTypeParmType *templateTypeParm;
    TemplateSymbol *functionTemplateSym;
    SimpleTemplateIDType *simpleTemplateID;
}


%token TOKEN_INT8_CONSTANT TOKEN_UINT8_CONSTANT
%token TOKEN_INT16_CONSTANT TOKEN_UINT16_CONSTANT
%token TOKEN_INT32_CONSTANT TOKEN_UINT32_CONSTANT
%token TOKEN_INT64_CONSTANT TOKEN_UINT64_CONSTANT
%token TOKEN_INT32DOTDOTDOT_CONSTANT TOKEN_UINT32DOTDOTDOT_CONSTANT
%token TOKEN_INT64DOTDOTDOT_CONSTANT TOKEN_UINT64DOTDOTDOT_CONSTANT
%token <stringVal> TOKEN_FLOAT16_CONSTANT
%token TOKEN_FLOAT_CONSTANT TOKEN_DOUBLE_CONSTANT TOKEN_STRING_C_LITERAL TOKEN_STRING_SYCL_LITERAL
%token <stringVal> TOKEN_IDENTIFIER TOKEN_STRING_LITERAL TOKEN_TYPE_NAME
%token TOKEN_PRAGMA TOKEN_NULL
%token <stringVal> TOKEN_TEMPLATE_NAME
%token TOKEN_TEMPLATE TOKEN_TYPENAME
%token TOKEN_PTR_OP TOKEN_INC_OP TOKEN_DEC_OP TOKEN_LEFT_OP TOKEN_RIGHT_OP
%token TOKEN_LE_OP TOKEN_GE_OP TOKEN_EQ_OP TOKEN_NE_OP
%token TOKEN_AND_OP TOKEN_OR_OP TOKEN_MUL_ASSIGN TOKEN_DIV_ASSIGN TOKEN_MOD_ASSIGN
%token TOKEN_ADD_ASSIGN TOKEN_SUB_ASSIGN TOKEN_LEFT_ASSIGN TOKEN_RIGHT_ASSIGN
%token TOKEN_AND_ASSIGN TOKEN_OR_ASSIGN TOKEN_XOR_ASSIGN
%token TOKEN_SIZEOF TOKEN_NEW TOKEN_DELETE TOKEN_IN TOKEN_ALLOCA
%token <stringVal> TOKEN_INTRINSIC_CALL

%token TOKEN_EXTERN TOKEN_EXPORT TOKEN_STATIC TOKEN_INLINE TOKEN_NOINLINE TOKEN_VECTORCALL TOKEN_REGCALL TOKEN_TASK TOKEN_DECLSPEC
%token TOKEN_UNIFORM TOKEN_VARYING TOKEN_TYPEDEF TOKEN_SOA TOKEN_UNMASKED
%token TOKEN_INT TOKEN_SIGNED TOKEN_UNSIGNED TOKEN_FLOAT16 TOKEN_FLOAT TOKEN_DOUBLE
%token TOKEN_INT8 TOKEN_INT16 TOKEN_INT64 TOKEN_CONST TOKEN_VOID TOKEN_BOOL
%token TOKEN_UINT8 TOKEN_UINT16 TOKEN_UINT TOKEN_UINT64
%token TOKEN_ENUM TOKEN_STRUCT TOKEN_TRUE TOKEN_FALSE

%token TOKEN_CASE TOKEN_DEFAULT TOKEN_IF TOKEN_ELSE TOKEN_SWITCH
%token TOKEN_WHILE TOKEN_DO TOKEN_LAUNCH TOKEN_FOREACH TOKEN_FOREACH_TILED
%token TOKEN_FOREACH_UNIQUE TOKEN_FOREACH_ACTIVE TOKEN_DOTDOTDOT
%token TOKEN_FOR TOKEN_GOTO TOKEN_CONTINUE TOKEN_BREAK TOKEN_RETURN
%token TOKEN_CIF TOKEN_CDO TOKEN_CFOR TOKEN_CWHILE
%token TOKEN_SYNC TOKEN_PRINT TOKEN_ASSERT TOKEN_INVOKE_SYCL

%type <expr> primary_expression postfix_expression integer_dotdotdot
%type <expr> unary_expression cast_expression funcall_expression launch_expression intrincall_expression
%type <expr> multiplicative_expression additive_expression shift_expression
%type <expr> relational_expression equality_expression and_expression
%type <expr> exclusive_or_expression inclusive_or_expression
%type <expr> invoke_sycl_expression
%type <expr> logical_and_expression logical_or_expression new_expression
%type <expr> conditional_expression assignment_expression expression
%type <expr> initializer constant_expression for_test
%type <exprList> argument_expression_list initializer_list

%type <stmt> attributed_statement labeled_statement compound_statement for_init_statement statement
%type <stmt> expression_statement selection_statement iteration_statement
%type <stmt> jump_statement statement_list declaration_statement print_statement
%type <stmt> assert_statement sync_statement delete_statement unmasked_statement

%type <declaration> declaration parameter_declaration
%type <declarators> init_declarator_list
%type <declarationList> parameter_list parameter_type_list
%type <declarator> declarator pointer reference
%type <declarator> init_declarator direct_declarator struct_declarator
%type <declarator> abstract_declarator direct_abstract_declarator

%type <structDeclaratorList> struct_declarator_list
%type <structDeclaration> struct_declaration
%type <structDeclarationList> struct_declaration_list

%type <symbolList> enumerator_list
%type <symbol> enumerator foreach_identifier foreach_active_identifier template_int_parameter template_enum_parameter
%type <enumType> enum_specifier

%type <type> specifier_qualifier_list struct_or_union_specifier
%type <type> struct_or_union_and_name
%type <type> type_specifier type_name rate_qualified_type_specifier
%type <type> short_vec_specifier
%type <typeList> type_specifier_list
%type <atomicType> atomic_var_type_specifier int_constant_type template_int_constant_type

%type <typeQualifier> type_qualifier type_qualifier_list
%type <storageClass> storage_class_specifier
%type <declSpecs> declaration_specifiers

%type <stringVal> string_constant intrinsic_name
%type <constCharPtr> struct_or_union_name enum_identifier goto_identifier
%type <constCharPtr> foreach_unique_identifier

%type <intVal> int_constant soa_width_specifier rate_qualified_new

%type <pragmaAttributes> pragma
%type <foreachDimension> foreach_dimension_specifier
%type <foreachDimensionList> foreach_dimension_list

%type <declspecPair> declspec_item
%type <declspecList> declspec_specifier declspec_list

%type <constCharPtr> template_identifier
%type <templateArg> template_argument
%type <templateArgs> template_argument_list
%type <simpleTemplateID> simple_template_id template_function_specialization_declaration
%type <templateTypeParm> template_type_parameter
%type <templateParm> template_parameter
%type <templateParmList> template_parameter_list template_head
%type <functionTemplateSym> template_declaration

%destructor { lCleanUpString($$); } <stringVal>
// TODO! destructos for all semantic types that return pointer to heap-allocated memory
// e.g., tests/lit-tests/2599.ispc

%start translation_unit
%%

string_constant
    : TOKEN_STRING_LITERAL
    {
        $$ = new std::string(*$1);
        lCleanUpString($1);
    }
    | string_constant TOKEN_STRING_LITERAL
    {
        std::string *p_str_cst = new std::string();
        p_str_cst->append(*$1);
        p_str_cst->append(*$2);
        $$ = p_str_cst;
        // Allocated in lStringConst
        lCleanUpString($1);
        lCleanUpString($2);
    }
    ;

primary_expression
    : TOKEN_IDENTIFIER {
        const char *name = $1->c_str();
        Symbol *s = m->symbolTable->LookupVariable(name);
        $$ = nullptr;
        if (s)
            $$ = new SymbolExpr(s, @1);
        else {
            std::vector<Symbol *> funs;
            m->symbolTable->LookupFunction(name, &funs);
            if (funs.size() > 0)
                $$ = new FunctionSymbolExpr(name, funs, @1);
        }
        if ($$ == nullptr) {
            std::vector<std::string> alternates =
                m->symbolTable->ClosestVariableOrFunctionMatch(name);
            std::string alts = lGetAlternates(alternates);
            Error(@1, "Undeclared symbol \"%s\".%s", name, alts.c_str());
        }
        lCleanUpString($1);
    }
    | TOKEN_INT8_CONSTANT {
        $$ = new ConstExpr(AtomicType::UniformInt8->GetAsConstType(),
                           (int8_t)yylval.intVal, @1);
    }
    | TOKEN_UINT8_CONSTANT {
        $$ = new ConstExpr(AtomicType::UniformUInt8->GetAsConstType(),
                           (uint8_t)yylval.intVal, @1);
    }
    | TOKEN_INT16_CONSTANT {
        $$ = new ConstExpr(AtomicType::UniformInt16->GetAsConstType(),
                           (int16_t)yylval.intVal, @1);
    }
    | TOKEN_UINT16_CONSTANT {
        $$ = new ConstExpr(AtomicType::UniformUInt16->GetAsConstType(),
                           (uint16_t)yylval.intVal, @1);
    }
    | TOKEN_INT32_CONSTANT {
        $$ = new ConstExpr(AtomicType::UniformInt32->GetAsConstType(),
                           (int32_t)yylval.intVal, @1);
    }
    | TOKEN_UINT32_CONSTANT {
        $$ = new ConstExpr(AtomicType::UniformUInt32->GetAsConstType(),
                           (uint32_t)yylval.intVal, @1);
    }
    | TOKEN_INT64_CONSTANT {
        $$ = new ConstExpr(AtomicType::UniformInt64->GetAsConstType(),
                           (int64_t)yylval.intVal, @1);
    }
    | TOKEN_UINT64_CONSTANT {
        $$ = new ConstExpr(AtomicType::UniformUInt64->GetAsConstType(),
                           (uint64_t)yylval.intVal, @1);
    }
    | TOKEN_FLOAT16_CONSTANT {
         std::string sval = *$1;
         lCleanUpString($1);
         llvm::Type *hType = llvm::Type::getHalfTy(*g->ctx);
         const llvm::fltSemantics &FS = hType->getFltSemantics();
         llvm::APFloat f16(FS, sval);
         $$ = new ConstExpr(AtomicType::UniformFloat16->GetAsConstType(),
                           f16, @1);
    }
    | TOKEN_FLOAT_CONSTANT {
        llvm::APFloat f(yylval.floatVal);
        $$ = new ConstExpr(AtomicType::UniformFloat->GetAsConstType(),
                           f, @1);
    }
    | TOKEN_DOUBLE_CONSTANT {
        llvm::APFloat d(yylval.doubleVal);
        $$ = new ConstExpr(AtomicType::UniformDouble->GetAsConstType(),
                           d, @1);
    }
    | TOKEN_TRUE {
        $$ = new ConstExpr(AtomicType::UniformBool->GetAsConstType(), true, @1);
    }
    | TOKEN_FALSE {
        $$ = new ConstExpr(AtomicType::UniformBool->GetAsConstType(), false, @1);
    }
    | TOKEN_NULL {
        $$ = new NullPointerExpr(@1);
    }
/*    | TOKEN_STRING_LITERAL
       { UNIMPLEMENTED }*/
    | '(' expression ')' { $$ = $2; }
    | '(' error ')' { $$ = nullptr; }
    ;

launch_expression
    : TOKEN_LAUNCH postfix_expression '(' argument_expression_list ')'
      {
          ConstExpr *oneExpr = new ConstExpr(AtomicType::UniformInt32, (int32_t)1, @2);
          Expr *launchCount[3] = {oneExpr, oneExpr, oneExpr};
          $$ = new FunctionCallExpr($2, $4, Union(@2, @5), true, launchCount);
      }
    | TOKEN_LAUNCH postfix_expression '(' ')'
      {
          ConstExpr *oneExpr = new ConstExpr(AtomicType::UniformInt32, (int32_t)1, @2);
          Expr *launchCount[3] = {oneExpr, oneExpr, oneExpr};
          $$ = new FunctionCallExpr($2, new ExprList(Union(@3,@4)), Union(@2, @4), true, launchCount);
       }

    | TOKEN_LAUNCH '[' assignment_expression ']' postfix_expression '(' argument_expression_list ')'
      {
          ConstExpr *oneExpr = new ConstExpr(AtomicType::UniformInt32, (int32_t)1, @5);
          Expr *launchCount[3] = {$3, oneExpr, oneExpr};
          $$ = new FunctionCallExpr($5, $7, Union(@5,@8), true, launchCount);
      }
    | TOKEN_LAUNCH '[' assignment_expression ']' postfix_expression '(' ')'
      {
          ConstExpr *oneExpr = new ConstExpr(AtomicType::UniformInt32, (int32_t)1, @5);
          Expr *launchCount[3] = {$3, oneExpr, oneExpr};
          $$ = new FunctionCallExpr($5, new ExprList(Union(@5,@6)), Union(@5,@7), true, launchCount);
      }

    | TOKEN_LAUNCH '[' assignment_expression ',' assignment_expression ']' postfix_expression '(' argument_expression_list ')'
      {
          ConstExpr *oneExpr = new ConstExpr(AtomicType::UniformInt32, (int32_t)1, @7);
          Expr *launchCount[3] = {$3, $5, oneExpr};
          $$ = new FunctionCallExpr($7, $9, Union(@7,@10), true, launchCount);
      }
    | TOKEN_LAUNCH '[' assignment_expression ',' assignment_expression ']' postfix_expression '(' ')'
      {
          ConstExpr *oneExpr = new ConstExpr(AtomicType::UniformInt32, (int32_t)1, @7);
          Expr *launchCount[3] = {$3, $5, oneExpr};
          $$ = new FunctionCallExpr($7, new ExprList(Union(@7,@8)), Union(@7,@9), true, launchCount);
      }
    | TOKEN_LAUNCH '[' assignment_expression ']' '[' assignment_expression ']' postfix_expression '(' argument_expression_list ')'
      {
          ConstExpr *oneExpr = new ConstExpr(AtomicType::UniformInt32, (int32_t)1, @8);
          Expr *launchCount[3] = {$6, $3, oneExpr};
          $$ = new FunctionCallExpr($8, $10, Union(@8,@11), true, launchCount);
      }
    | TOKEN_LAUNCH '[' assignment_expression ']' '[' assignment_expression ']' postfix_expression '(' ')'
      {
          ConstExpr *oneExpr = new ConstExpr(AtomicType::UniformInt32, (int32_t)1, @8);
          Expr *launchCount[3] = {$6, $3, oneExpr};
          $$ = new FunctionCallExpr($8, new ExprList(Union(@8,@9)), Union(@8,@10), true, launchCount);
      }

    | TOKEN_LAUNCH '[' assignment_expression ',' assignment_expression ',' assignment_expression ']' postfix_expression '(' argument_expression_list ')'
      {
          Expr *launchCount[3] = {$3, $5, $7};
          $$ = new FunctionCallExpr($9, $11, Union(@9,@12), true, launchCount);
      }
    | TOKEN_LAUNCH '[' assignment_expression ',' assignment_expression ',' assignment_expression ']' postfix_expression '(' ')'
      {
          Expr *launchCount[3] = {$3, $5, $7};
          $$ = new FunctionCallExpr($9, new ExprList(Union(@9,@10)), Union(@9,@11), true, launchCount);
      }
    | TOKEN_LAUNCH '[' assignment_expression ']' '[' assignment_expression ']' '[' assignment_expression ']' postfix_expression '(' argument_expression_list ')'
      {
          Expr *launchCount[3] = {$9, $6, $3};
          $$ = new FunctionCallExpr($11, $13, Union(@11,@14), true, launchCount);
      }
    | TOKEN_LAUNCH '[' assignment_expression ']' '[' assignment_expression ']' '[' assignment_expression ']' postfix_expression '(' ')'
      {
          Expr *launchCount[3] = {$9, $6, $3};
          $$ = new FunctionCallExpr($11, new ExprList(Union(@11,@12)), Union(@11,@13), true, launchCount);
      }


    | TOKEN_LAUNCH '<' postfix_expression '(' argument_expression_list ')' '>'
       {
          Error(Union(@2, @7), "\"launch\" expressions no longer take '<' '>' "
                "around function call expression.");
          $$ = nullptr;
       }
    | TOKEN_LAUNCH '<' postfix_expression '(' ')' '>'
       {
          Error(Union(@2, @6), "\"launch\" expressions no longer take '<' '>' "
                "around function call expression.");
          $$ = nullptr;
       }
    | TOKEN_LAUNCH '[' assignment_expression ']' '<' postfix_expression '(' argument_expression_list ')' '>'
       {
          Error(Union(@5, @10), "\"launch\" expressions no longer take '<' '>' "
                "around function call expression.");
          $$ = nullptr;
       }
    | TOKEN_LAUNCH '[' assignment_expression ']' '<' postfix_expression '(' ')' '>'
       {
          Error(Union(@5, @9), "\"launch\" expressions no longer take '<' '>' "
                "around function call expression.");
          $$ = nullptr;
       }
    ;

invoke_sycl_expression
    : TOKEN_INVOKE_SYCL '(' postfix_expression ')'
      {
          $$ = new FunctionCallExpr($3, new ExprList(@4), Union(@1,@4), false, nullptr, true);
      }
    | TOKEN_INVOKE_SYCL '(' postfix_expression ',' argument_expression_list ')'
      {
          $$ = new FunctionCallExpr($3, $5, Union(@1,@6), false, nullptr, true);
      }
    | TOKEN_INVOKE_SYCL '(' error ')'
      { $$ = nullptr; }
    ;

postfix_expression
    : primary_expression
    | postfix_expression '[' expression ']'
      { $$ = new IndexExpr($1, $3, Union(@1,@4)); }
    | postfix_expression '[' error ']'
      { $$ = nullptr; }
    | launch_expression
    | postfix_expression '.' TOKEN_IDENTIFIER
      {
          $$ = MemberExpr::create($1, yytext, Union(@1,@3), @3, false);
          lCleanUpString($3);
      }
    /* When we have postfix_expression inside template definition, we need to allow cases when
       member name equals to template name or template parameter name. */
    | postfix_expression '.' TOKEN_TYPE_NAME
      {
          $$ = MemberExpr::create($1, yytext, Union(@1,@3), @3, false);
          lCleanUpString($3);
      }
    | postfix_expression '.' TOKEN_TEMPLATE_NAME
      {
          $$ = MemberExpr::create($1, yytext, Union(@1,@3), @3, false);
          lCleanUpString($3);
      }
    | postfix_expression TOKEN_PTR_OP TOKEN_IDENTIFIER
      {
          $$ = MemberExpr::create($1, yytext, Union(@1,@3), @3, true);
          lCleanUpString($3);
      }
    /* When we have postfix_expression inside template definition, we need to allow cases when
       member name equals to template name or template parameter name. */
    | postfix_expression TOKEN_PTR_OP TOKEN_TYPE_NAME
      {
          $$ = MemberExpr::create($1, yytext, Union(@1,@3), @3, true);
          lCleanUpString($3);
      }
    | postfix_expression TOKEN_PTR_OP TOKEN_TEMPLATE_NAME
      {
          $$ = MemberExpr::create($1, yytext, Union(@1,@3), @3, true);
          lCleanUpString($3);
      }
    | postfix_expression TOKEN_INC_OP
      { $$ = new UnaryExpr(UnaryExpr::PostInc, $1, Union(@1,@2)); }
    | postfix_expression TOKEN_DEC_OP
      { $$ = new UnaryExpr(UnaryExpr::PostDec, $1, Union(@1,@2)); }
    ;

intrinsic_name
    : TOKEN_INTRINSIC_CALL
      {
          $$ = $1;
      }
    ;

intrincall_expression
    : intrinsic_name '(' argument_expression_list ')'
      {
          std::string *name = $1;
          name->erase(0, 1);
          Symbol* sym = m->AddLLVMIntrinsicDecl(*name, $3, Union(@1,@4));
          const char *fname = name->c_str();
          const std::vector<Symbol *> funcs{sym};
          FunctionSymbolExpr *fSym = nullptr;
          if (sym != nullptr)
              fSym = new FunctionSymbolExpr(fname, funcs, @1);
          $$ = new FunctionCallExpr(fSym, $3, Union(@1,@4));
          delete name;
      }
    ;

funcall_expression
    : postfix_expression
    | postfix_expression '(' ')'
      { $$ = new FunctionCallExpr($1, new ExprList(Union(@1,@2)), Union(@1,@3)); }
    | postfix_expression '(' argument_expression_list ')'
      { $$ = new FunctionCallExpr($1, $3, Union(@1,@4)); }
    | postfix_expression '(' error ')'
      { $$ = nullptr; }
    | simple_template_id '(' ')'
      {
          // Create FunctionSymbolExpr with a candidate functions list
          Expr *functionSymbolExpr = nullptr;
          std::vector<TemplateSymbol *> funcTempls;
          const std::string name = $1->first->name;
          m->symbolTable->LookupFunctionTemplate(name, &funcTempls);
          if (funcTempls.size() > 0) {
              TemplateArgs *templArgs = $1->second;
              Assert(templArgs);
              functionSymbolExpr = new FunctionSymbolExpr(name.c_str(), funcTempls, *templArgs, @1);
              $$ = new FunctionCallExpr(functionSymbolExpr, new ExprList(Union(@1,@2)), Union(@1,@3));
          } else {
              Error(@1, "No matching template functions were declared.");
              $$ = nullptr;
          }

          // deallocate SimpleTemplateIDType returned by simple_template_id
          lFreeSimpleTemplateID($1);
      }
    | simple_template_id '(' argument_expression_list ')'
      {
          // Create FunctionSymbolExpr with a candidate functions list
          Expr *functionSymbolExpr = nullptr;
          std::vector<TemplateSymbol *> funcTempls;
          const std::string name = $1->first->name;
          m->symbolTable->LookupFunctionTemplate(name, &funcTempls);
          if (funcTempls.size() > 0) {
              TemplateArgs *templArgs = $1->second;
              Assert(templArgs);
              functionSymbolExpr = new FunctionSymbolExpr(name.c_str(), funcTempls, *templArgs, @1);
              $$ = new FunctionCallExpr(functionSymbolExpr, $3, Union(@1,@4));
          } else {
              Error(@1, "No matching template functions were declared.");
              $$ = nullptr;
          }

          // deallocate SimpleTemplateIDType returned by simple_template_id
          lFreeSimpleTemplateID($1);
      }
    | simple_template_id '(' error ')'
      {
          $$ = nullptr;

          // deallocate SimpleTemplateIDType returned by simple_template_id
          lFreeSimpleTemplateID($1);
      }
    ;

argument_expression_list
    : assignment_expression      { $$ = new ExprList($1, @1); }
    | argument_expression_list ',' assignment_expression
      {
          ExprList *argList = llvm::dyn_cast<ExprList>($1);
          if (argList == nullptr) {
              AssertPos(@1, m->errorCount > 0);
              argList = new ExprList(@3);
          }
          argList->exprs.push_back($3);
          argList->pos = Union(argList->pos, @3);
          $$ = argList;
      }
    ;

unary_expression
    : funcall_expression
    | intrincall_expression
    | invoke_sycl_expression
    | TOKEN_INC_OP unary_expression
      { $$ = new UnaryExpr(UnaryExpr::PreInc, $2, Union(@1, @2)); }
    | TOKEN_DEC_OP unary_expression
      { $$ = new UnaryExpr(UnaryExpr::PreDec, $2, Union(@1, @2)); }
    | '&' unary_expression
      { $$ = new AddressOfExpr($2, Union(@1, @2)); }
    | '*' unary_expression
      { $$ = new PtrDerefExpr($2, Union(@1, @2)); }
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
    | TOKEN_ALLOCA '(' assignment_expression ')'
      {
          $$ = new AllocaExpr($3, Union(@1, @4));
      }
    ;

cast_expression
    : unary_expression
    | '(' type_name ')' cast_expression
      {
          $$ = new TypeCastExpr($2, $4, Union(@1,@4));
      }
    ;

multiplicative_expression
    : cast_expression
    | multiplicative_expression '*' cast_expression
      { $$ = MakeBinaryExpr(BinaryExpr::Mul, $1, $3, Union(@1, @3)); }
    | multiplicative_expression '/' cast_expression
      { $$ = MakeBinaryExpr(BinaryExpr::Div, $1, $3, Union(@1, @3)); }
    | multiplicative_expression '%' cast_expression
      { $$ = MakeBinaryExpr(BinaryExpr::Mod, $1, $3, Union(@1, @3)); }
    ;

additive_expression
    : multiplicative_expression
    | additive_expression '+' multiplicative_expression
      { $$ = MakeBinaryExpr(BinaryExpr::Add, $1, $3, Union(@1, @3)); }
    | additive_expression '-' multiplicative_expression
      { $$ = MakeBinaryExpr(BinaryExpr::Sub, $1, $3, Union(@1, @3)); }
    ;

shift_expression
    : additive_expression
    | shift_expression TOKEN_LEFT_OP additive_expression
      { $$ = MakeBinaryExpr(BinaryExpr::Shl, $1, $3, Union(@1, @3)); }
    | shift_expression TOKEN_RIGHT_OP additive_expression
      { $$ = MakeBinaryExpr(BinaryExpr::Shr, $1, $3, Union(@1, @3)); }
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

rate_qualified_new
    : TOKEN_NEW { $$ = 0; }
    | TOKEN_UNIFORM TOKEN_NEW { $$ = TYPEQUAL_UNIFORM; }
    | TOKEN_VARYING TOKEN_NEW { $$ = TYPEQUAL_VARYING; }
    ;

rate_qualified_type_specifier
    : type_specifier { $$ = $1; }
    | TOKEN_UNIFORM type_specifier
    {
        if ($2 == nullptr)
            $$ = nullptr;
        else if ($2->IsVoidType()) {
            Error(@1, "\"uniform\" qualifier is illegal with \"void\" type.");
            $$ = nullptr;
        }
        else
            $$ = $2->GetAsUniformType();
    }
    | TOKEN_VARYING type_specifier
    {
        if ($2 == nullptr)
            $$ = nullptr;
        else if ($2->IsVoidType()) {
            Error(@1, "\"varying\" qualifier is illegal with \"void\" type.");
            $$ = nullptr;
        }
        else
            $$ = $2->GetAsVaryingType();
    }
    | soa_width_specifier type_specifier
    {
        if ($2 == nullptr)
            $$ = nullptr;
        else {
            int soaWidth = (int)$1;
            const StructType *st = CastType<StructType>($2);
            if (st == nullptr) {
                Error(@1, "\"soa\" qualifier is illegal with non-struct type \"%s\".",
                      $2->GetString().c_str());
                $$ = nullptr;
            }
            else if (soaWidth <= 0 || (soaWidth & (soaWidth - 1)) != 0) {
                Error(@1, "soa<%d> width illegal. Value must be positive power "
                      "of two.", soaWidth);
                $$ = nullptr;
            }
            else
                $$ = st->GetAsSOAType(soaWidth);
        }
    }
    ;

new_expression
    : conditional_expression
    | rate_qualified_new rate_qualified_type_specifier
    {
        $$ = new NewExpr((int32_t)$1, $2, nullptr, nullptr, @1, Union(@1, @2));
    }
    | rate_qualified_new rate_qualified_type_specifier '(' initializer_list ')'
    {
        $$ = new NewExpr((int32_t)$1, $2, $4, nullptr, @1, Union(@1, @2));
    }
    | rate_qualified_new rate_qualified_type_specifier '[' expression ']'
    {
        $$ = new NewExpr((int32_t)$1, $2, nullptr, $4, @1, Union(@1, @4));
    }
    ;

assignment_expression
    : new_expression
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
        if ($1 == nullptr) {
            AssertPos(@1, m->errorCount > 0);
            $$ = nullptr;
        }
        else if ($1->declSpecs->storageClass == SC_TYPEDEF) {
            for (unsigned int i = 0; i < $1->declarators.size(); ++i) {
                if ($1->declarators[i] == nullptr)
                    AssertPos(@1, m->errorCount > 0);
                else
                    m->AddTypeDef($1->declarators[i]->name,
                                  $1->declarators[i]->type,
                                  $1->declarators[i]->pos);
            }
            $$ = nullptr;
        }
        else {
            $1->DeclareFunctions();
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
          // init_declarator_list returns vector of declarators, its copy is
          // saved in Declaration constructor, so it is not needed anymore.
          delete $2;
      }
    ;

soa_width_specifier
    : TOKEN_SOA '<' int_constant '>'
      { $$ = $3; }
    ;

declspec_item
    : TOKEN_IDENTIFIER
    {
        std::pair<std::string, SourcePos> *p = new std::pair<std::string, SourcePos>;
        p->first = *$1;
        p->second = @1;
        $$ = p;
        lCleanUpString($1);
    }
    ;

declspec_list
    : declspec_item
    {
        $$ = new std::vector<std::pair<std::string, SourcePos> >;
        $$->push_back(*$1);
        // declspec_item returns pair that was copied so it is not needed anymore.
        delete $1;
    }
    | declspec_list ',' declspec_item
    {
        if ($1 != nullptr) {
            $1->push_back(*$3);
            // declspec_item returns pair that was copied so it is not needed anymore.
            delete $3;
        }
        $$ = $1;
    }
    ;

declspec_specifier
    : TOKEN_DECLSPEC '(' declspec_list ')'
    {
        // declspec_list returns heap allocated vector that passed up here.
        $$ = $3;
    }
    ;

declaration_specifiers
    : storage_class_specifier
      {
          $$ = new DeclSpecs(nullptr, $1);
      }
    | storage_class_specifier declaration_specifiers
      {
          DeclSpecs *ds = (DeclSpecs *)$2;
          if (ds != nullptr) {
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
    | declspec_specifier
      {
          $$ = new DeclSpecs;
          if ($1 != nullptr) {
              $$->declSpecList = *$1;
              // declspec_specifier returns a vector that was copied and it is not needed anymore.
              delete $1;
          }
      }
    | declspec_specifier declaration_specifiers
      {
          DeclSpecs *ds = (DeclSpecs *)$2;
          std::vector<std::pair<std::string, SourcePos> > *declSpecList = $1;
          if (ds != nullptr && declSpecList != nullptr) {
              for (int i = 0; i < (int)declSpecList->size(); ++i)
                  ds->declSpecList.push_back((*declSpecList)[i]);

              // declspec_specifier returns a vector that was copied and it is not needed anymore.
              delete declSpecList;
          }
          $$ = ds;
      }
    | soa_width_specifier
      {
          DeclSpecs *ds = new DeclSpecs;
          ds->soaWidth = (int32_t)$1;
          $$ = ds;
      }
    | soa_width_specifier declaration_specifiers
      {
          DeclSpecs *ds = (DeclSpecs *)$2;
          if (ds != nullptr) {
              if (ds->soaWidth != 0)
                  Error(@1, "soa<> qualifier supplied multiple times in declaration.");
              else
                  ds->soaWidth = (int32_t)$1;
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
          ds->vectorSize = (int32_t)$3;
          $$ = ds;
    }
    | type_specifier declaration_specifiers
      {
          DeclSpecs *ds = (DeclSpecs *)$2;
          if (ds != nullptr) {
              if (ds->baseType != nullptr) {
                  if( ds->baseType->IsUnsignedType()) {
                      Error(@1, "Redefining uint8/uint16/uint32/uint64 type "
                      "which is part of ISPC language since version 1.13. "
                      "Remove this typedef or use ISPC_UINT_IS_DEFINED to "
                      "detect that these types are defined.");
                  }
                  else
                      Error(@1, "Multiple types provided for declaration.");
              }
              ds->baseType = $1;
          }
          $$ = ds;
      }
    | type_qualifier
      {
          $$ = new DeclSpecs(nullptr, SC_NONE, $1);
      }
    | type_qualifier declaration_specifiers
      {
          DeclSpecs *ds = (DeclSpecs *)$2;
          if (ds != nullptr)
              ds->typeQualifiers |= $1;
          $$ = ds;
      }
    ;

init_declarator_list
    : init_declarator
      {
          std::vector<Declarator *> *dl = new std::vector<Declarator *>;
          if ($1 != nullptr)
              dl->push_back($1);
          $$ = dl;
      }
    | init_declarator_list ',' init_declarator
      {
          std::vector<Declarator *> *dl = (std::vector<Declarator *> *)$1;
          if (dl == nullptr) {
              AssertPos(@1, m->errorCount > 0);
              dl = new std::vector<Declarator *>;
          }
          if ($3 != nullptr)
              dl->push_back($3);
          $$ = dl;
      }
    ;

init_declarator
    : declarator
    | declarator '=' initializer
      {
          if ($1 != nullptr)
              $1->initExpr = $3;
          $$ = $1;
      }
    ;

storage_class_specifier
    : TOKEN_TYPEDEF { $$ = SC_TYPEDEF; }
    | TOKEN_EXTERN { $$ = SC_EXTERN; }
    | TOKEN_EXTERN TOKEN_STRING_C_LITERAL  { $$ = SC_EXTERN_C; }
    | TOKEN_EXTERN TOKEN_STRING_SYCL_LITERAL  { $$ = SC_EXTERN_SYCL; }
    | TOKEN_STATIC { $$ = SC_STATIC; }
    ;

type_specifier
    : atomic_var_type_specifier { $$ = $1; }
    | TOKEN_TYPE_NAME
    {
        const Type *t = m->symbolTable->LookupType(yytext);
        $$ = t;
        lCleanUpString($1);
    }
    | struct_or_union_specifier { $$ = $1; }
    | enum_specifier { $$ = $1; }
    ;

type_specifier_list
    : type_specifier
    {
        if ($1 == nullptr)
            $$ = nullptr;
        else {
            std::vector<std::pair<const Type *, SourcePos> > *vec =
                new std::vector<std::pair<const Type *, SourcePos> >;
            vec->push_back(std::make_pair($1, @1));
            $$ = vec;
        }
    }
    | type_specifier_list ',' type_specifier
    {
        $$ = $1;
        if ($1 == nullptr)
            Assert(m->errorCount > 0);
        else
            $$->push_back(std::make_pair($3, @3));
    }
    ;

atomic_var_type_specifier
    : TOKEN_VOID { $$ = AtomicType::Void; }
    | TOKEN_BOOL { $$ = AtomicType::UniformBool->GetAsUnboundVariabilityType(); }
    | TOKEN_INT8 { $$ = AtomicType::UniformInt8->GetAsUnboundVariabilityType(); }
    | TOKEN_UINT8 { $$ = AtomicType::UniformUInt8->GetAsUnboundVariabilityType(); }
    | TOKEN_INT16 { $$ = AtomicType::UniformInt16->GetAsUnboundVariabilityType(); }
    | TOKEN_UINT16 { $$ = AtomicType::UniformUInt16->GetAsUnboundVariabilityType(); }
    | TOKEN_INT { $$ = AtomicType::UniformInt32->GetAsUnboundVariabilityType(); }
    | TOKEN_UINT { $$ = AtomicType::UniformUInt32->GetAsUnboundVariabilityType(); }
    | TOKEN_FLOAT16 { $$ = AtomicType::UniformFloat16->GetAsUnboundVariabilityType(); }
    | TOKEN_FLOAT { $$ = AtomicType::UniformFloat->GetAsUnboundVariabilityType(); }
    | TOKEN_DOUBLE { $$ = AtomicType::UniformDouble->GetAsUnboundVariabilityType(); }
    | TOKEN_INT64 { $$ = AtomicType::UniformInt64->GetAsUnboundVariabilityType(); }
    | TOKEN_UINT64 { $$ = AtomicType::UniformUInt64->GetAsUnboundVariabilityType(); }
    ;

short_vec_specifier
    : atomic_var_type_specifier '<' int_constant '>'
    {
        $$ = $1 ? new VectorType($1, (int32_t)$3) : nullptr;
    }
    ;

struct_or_union_name
    : TOKEN_IDENTIFIER
    {
        $$ = strdup(yytext);
        lCleanUpString($1);
    }
    | TOKEN_TYPE_NAME
    {
        $$ = strdup(yytext);
        lCleanUpString($1);
    }
    ;

struct_or_union_and_name
    : struct_or_union struct_or_union_name
      {
          const Type *st = m->symbolTable->LookupType($2);
          if (st == nullptr) {
              st = new UndefinedStructType($2, Variability::Unbound, false, @2);
              m->symbolTable->AddType($2, st, @2);
              $$ = st;
          }
          else {
              if (CastType<StructType>(st) == nullptr &&
                  CastType<UndefinedStructType>(st) == nullptr) {
                  Error(@2, "Type \"%s\" is not a struct type! (%s)", $2,
                        st->GetString().c_str());
                  $$ = nullptr;
              }
              else
                  $$ = st;
         }
         // allocated by strdup in struct_or_union_name
         free((char*)$2);
      }
    ;

struct_or_union_specifier
    : struct_or_union_and_name
    | struct_or_union_and_name '{' struct_declaration_list '}'
      {
          if ($3 != nullptr) {
              llvm::SmallVector<const Type *, 8> elementTypes;
              llvm::SmallVector<std::string, 8> elementNames;
              llvm::SmallVector<SourcePos, 8> elementPositions;
              GetStructTypesNamesPositions(*$3, &elementTypes, &elementNames,
                                           &elementPositions);
              const std::string &name = CastType<StructType>($1) ?
                  CastType<StructType>($1)->GetStructName() :
                  CastType<UndefinedStructType>($1)->GetStructName();
              StructType *st = new StructType(name, elementTypes, elementNames,
                                              elementPositions, false,
                                              Variability::Unbound, false, @1);
              m->symbolTable->AddType(name.c_str(), st, @1);
              $$ = st;
              // struct_declaration_list returns a vector that is not needed anymore.
              delete $3;
          }
          else
              $$ = nullptr;
      }
    | struct_or_union '{' struct_declaration_list '}'
      {
          if ($3 != nullptr) {
              llvm::SmallVector<const Type *, 8> elementTypes;
              llvm::SmallVector<std::string, 8> elementNames;
              llvm::SmallVector<SourcePos, 8> elementPositions;
              GetStructTypesNamesPositions(*$3, &elementTypes, &elementNames,
                                           &elementPositions);
              $$ = new StructType("", elementTypes, elementNames, elementPositions,
                                  false, Variability::Unbound, true, @1);
              // struct_declaration_list returns a vector that is not needed anymore.
              delete $3;
          }
          else
              $$ = nullptr;
      }
    | struct_or_union '{' '}'
      {
          llvm::SmallVector<const Type *, 8> elementTypes;
          llvm::SmallVector<std::string, 8> elementNames;
          llvm::SmallVector<SourcePos, 8> elementPositions;
          $$ = new StructType("", elementTypes, elementNames, elementPositions,
                              false, Variability::Unbound, true, @1);
      }
    | struct_or_union_and_name '{' '}'
      {
          llvm::SmallVector<const Type *, 8> elementTypes;
          llvm::SmallVector<std::string, 8> elementNames;
          llvm::SmallVector<SourcePos, 8> elementPositions;
          const std::string &name = CastType<StructType>($1) ?
              CastType<StructType>($1)->GetStructName() :
              CastType<UndefinedStructType>($1)->GetStructName();
          StructType *st = new StructType(name, elementTypes,
                                          elementNames, elementPositions,
                                          false, Variability::Unbound, false, @1);
          m->symbolTable->AddType(name.c_str(), st, @2);
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
          if ($1 != nullptr)
              sdl->push_back($1);
          $$ = sdl;
      }
    | struct_declaration_list struct_declaration
      {
          std::vector<StructDeclaration *> *sdl = (std::vector<StructDeclaration *> *)$1;
          if (sdl == nullptr) {
              AssertPos(@1, m->errorCount > 0);
              sdl = new std::vector<StructDeclaration *>;
          }
          if ($2 != nullptr)
              sdl->push_back($2);
          $$ = sdl;
      }
    ;

struct_declaration
    : specifier_qualifier_list struct_declarator_list ';'
      { $$ = ($1 != nullptr && $2 != nullptr) ? new StructDeclaration($1, $2) : nullptr; }
    ;

specifier_qualifier_list
    : type_specifier specifier_qualifier_list
    | type_specifier
    | short_vec_specifier
    | type_qualifier specifier_qualifier_list
    {
        if ($2 != nullptr) {
            if ($1 == TYPEQUAL_UNIFORM) {
                if ($2->IsVoidType()) {
                    Error(@1, "\"uniform\" qualifier is illegal with \"void\" type.");
                    $$ = nullptr;
                }
                else
                    $$ = $2->GetAsUniformType();
            }
            else if ($1 == TYPEQUAL_VARYING) {
                if ($2->IsVoidType()) {
                    Error(@1, "\"varying\" qualifier is illegal with \"void\" type.");
                    $$ = nullptr;
                }
                else
                    $$ = $2->GetAsVaryingType();
            }
            else if ($1 == TYPEQUAL_CONST)
                $$ = $2->GetAsConstType();
            else if ($1 == TYPEQUAL_SIGNED) {
                if ($2->IsIntType() == false) {
                    Error(@1, "Can't apply \"signed\" qualifier to \"%s\" type.",
                          $2->ResolveUnboundVariability(Variability::Varying)->GetString().c_str());
                    $$ = $2;
                }
            }
            else if ($1 == TYPEQUAL_UNSIGNED) {
                const Type *t = $2->GetAsUnsignedType();
                if (t)
                    $$ = t;
                else {
                    Error(@1, "Can't apply \"unsigned\" qualifier to \"%s\" type. Ignoring.",
                          $2->ResolveUnboundVariability(Variability::Varying)->GetString().c_str());
                    $$ = $2;
                }
            }
            else if ($1 == TYPEQUAL_INLINE) {
                Error(@1, "\"inline\" qualifier is illegal outside of "
                      "function declarations.");
                $$ = $2;
            }
            else if ($1 == TYPEQUAL_NOINLINE) {
                Error(@1, "\"noinline\" qualifier is illegal outside of "
                      "function declarations.");
                $$ = $2;
            }
            else if ($1 == TYPEQUAL_VECTORCALL) {
                Error(@1, "\"__vectorcall\" qualifier is illegal outside of "
                      "function declarations.");
                $$ = $2;
            }
            else if ($1 == TYPEQUAL_REGCALL) {
                Error(@1, "\"__regcall\" qualifier is illegal outside of "
                      "function declarations.");
                $$ = $2;
            }
            else if ($1 == TYPEQUAL_TASK) {
                Error(@1, "\"task\" qualifier is illegal outside of "
                      "function declarations.");
                $$ = $2;
            }
            else if ($1 == TYPEQUAL_UNMASKED) {
                Error(@1, "\"unmasked\" qualifier is illegal outside of "
                      "function declarations.");
                $$ = $2;
            }
            else if ($1 == TYPEQUAL_EXPORT) {
                Error(@1, "\"export\" qualifier is illegal outside of "
                      "function declarations.");
                $$ = $2;
            }
            else
                FATAL("Unhandled type qualifier in parser.");
        }
        else {
            if (m->errorCount == 0)
                Error(@1, "Lost type qualifier in parser.");
            $$ = nullptr;
        }
    }
    ;


struct_declarator_list
    : struct_declarator
      {
          std::vector<Declarator *> *sdl = new std::vector<Declarator *>;
          if ($1 != nullptr)
              sdl->push_back($1);
          $$ = sdl;
      }
    | struct_declarator_list ',' struct_declarator
      {
          std::vector<Declarator *> *sdl = (std::vector<Declarator *> *)$1;
          if (sdl == nullptr) {
              AssertPos(@1, m->errorCount > 0);
              sdl = new std::vector<Declarator *>;
          }
          if ($3 != nullptr)
              sdl->push_back($3);
          $$ = sdl;
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
    : TOKEN_IDENTIFIER
      {
          $$ = strdup(yytext);
          lCleanUpString($1);
      }
    ;

enum_specifier
    : TOKEN_ENUM '{' enumerator_list '}'
      {
          $$ = lCreateEnumType(nullptr, $3, @1);
          // enumerator_list returns aux vector that is not needed anymore.
          delete $3;
      }
    | TOKEN_ENUM enum_identifier '{' enumerator_list '}'
      {
          $$ = lCreateEnumType($2, $4, @2);
          // allocated by strdup in enum_identifier
          free((char*)$2);

          // enumerator_list returns aux vector that is not needed anymore.
          delete $4;
      }
    | TOKEN_ENUM '{' enumerator_list ',' '}'
      {
          $$ = lCreateEnumType(nullptr, $3, @1);
          // enumerator_list returns aux vector that is not needed anymore.
          delete $3;
      }
    | TOKEN_ENUM enum_identifier '{' enumerator_list ',' '}'
      {
          $$ = lCreateEnumType($2, $4, @2);
          // allocated by strdup in enum_identifier
          free((char*)$2);

          // enumerator_list returns aux vector that is not needed anymore.
          delete $4;
      }
    | TOKEN_ENUM enum_identifier
      {
          const Type *type = m->symbolTable->LookupType($2);
          if (type == nullptr) {
              std::vector<std::string> alternates = m->symbolTable->ClosestEnumTypeMatch($2);
              std::string alts = lGetAlternates(alternates);
              Error(@2, "Enum type \"%s\" unknown.%s", $2, alts.c_str());
              $$ = nullptr;
          }
          else {
              const EnumType *enumType = CastType<EnumType>(type);
              if (enumType == nullptr) {
                  Error(@2, "Type \"%s\" is not an enum type (%s).", $2,
                        type->GetString().c_str());
                  $$ = nullptr;
              }
              else
                  $$ = enumType;
          }
          // allocated by strdup in enum_identifier
          free((char*)$2);
      }
    ;

enumerator_list
    : enumerator
      {
          if ($1 == nullptr)
              $$ = nullptr;
          else {
              std::vector<Symbol *> *el = new std::vector<Symbol *>;
              el->push_back($1);
              $$ = el;
          }
      }
    | enumerator_list ',' enumerator
      {
          std::vector<Symbol *> *symList = $1;
          if (symList == nullptr) {
              AssertPos(@1, m->errorCount > 0);
              symList = new std::vector<Symbol *>;
          }
          if ($3 != nullptr)
              symList->push_back($3);
          $$ = symList;
      }
    ;

enumerator
    : enum_identifier
      {
          $$ = new Symbol($1, @1);
          // allocated by strdup in enum_identifier
          free((char*)$1);
      }
    | enum_identifier '=' constant_expression
      {
          int value;
          if ($1 != nullptr && $3 != nullptr &&
              lGetConstantInt($3, &value, @3, "Enumerator value")) {
              Symbol *sym = new Symbol($1, @1);
              sym->constValue = new ConstExpr(AtomicType::UniformUInt32->GetAsConstType(),
                                              (uint32_t)value, @3);
              $$ = sym;
          }
          else
              $$ = nullptr;

          // allocated by strdup in enum_identifier
          free((char*)$1);
      }
    ;

type_qualifier
    : TOKEN_CONST         { $$ = TYPEQUAL_CONST; }
    | TOKEN_UNIFORM       { $$ = TYPEQUAL_UNIFORM; }
    | TOKEN_VARYING       { $$ = TYPEQUAL_VARYING; }
    | TOKEN_TASK          { $$ = TYPEQUAL_TASK; }
    | TOKEN_UNMASKED      { $$ = TYPEQUAL_UNMASKED; }
    | TOKEN_EXPORT        { $$ = TYPEQUAL_EXPORT; }
    | TOKEN_INLINE        { $$ = TYPEQUAL_INLINE; }
    | TOKEN_NOINLINE      { $$ = TYPEQUAL_NOINLINE; }
    | TOKEN_VECTORCALL    { $$ = TYPEQUAL_VECTORCALL; }
    | TOKEN_REGCALL       { $$ = TYPEQUAL_REGCALL; }
    | TOKEN_SIGNED        { $$ = TYPEQUAL_SIGNED; }
    | TOKEN_UNSIGNED      { $$ = TYPEQUAL_UNSIGNED; }
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
        if ($1 != nullptr) {
            Declarator *tail = $1;
            while (tail->child != nullptr)
               tail = tail->child;
            tail->child = $2;
            $$ = $1;
        }
        else
            $$ = nullptr;
    }
    | reference direct_declarator
    {
        if ($1 != nullptr) {
            Declarator *tail = $1;
            while (tail->child != nullptr)
               tail = tail->child;
            tail->child = $2;
            $$ = $1;
        }
        else
            $$ = nullptr;
    }
    | direct_declarator
    ;

int_constant
    : TOKEN_INT8_CONSTANT { $$ = yylval.intVal; }
    | TOKEN_INT16_CONSTANT { $$ = yylval.intVal; }
    | TOKEN_INT32_CONSTANT { $$ = yylval.intVal; }
    | TOKEN_INT64_CONSTANT { $$ = yylval.intVal; }
    ;

direct_declarator
    : TOKEN_IDENTIFIER
      {
          Declarator *d = new Declarator(DK_BASE, @1);
          d->name = yytext;
          $$ = d;
          lCleanUpString($1);
      }
    // For the purpose of declaration, template_name token is no different from identifier token,
    // it needs to be processed in the same way. Semantic checks will be done later.
    | TOKEN_TEMPLATE_NAME
      {
          Declarator *d = new Declarator(DK_BASE, @1);
          d->name = yytext;
          $$ = d;
          lCleanUpString($1);
      }
    | '(' declarator ')'
    {
        $$ = $2;
    }
    | direct_declarator '[' constant_expression ']'
    {
        int size;
        if ($1 != nullptr && lGetConstantInt($3, &size, @3, "Array dimension")) {
            if (size < 0) {
                Error(@3, "Array dimension must be non-negative.");
                $$ = nullptr;
            }
            else {
                Declarator *d = new Declarator(DK_ARRAY, Union(@1, @4));
                d->arraySize = size;
                d->child = $1;
                $$ = d;
            }
        }
        else
            $$ = nullptr;
    }
    | direct_declarator '[' ']'
    {
        if ($1 != nullptr) {
            Declarator *d = new Declarator(DK_ARRAY, Union(@1, @3));
            d->arraySize = 0; // unsize
            d->child = $1;
            $$ = d;
        }
        else
            $$ = nullptr;
    }
    | direct_declarator '[' error ']'
    {
         $$ = nullptr;
    }
    | direct_declarator '(' parameter_type_list ')'
      {
          if ($1 != nullptr) {
              Declarator *d = new Declarator(DK_FUNCTION, Union(@1, @4));
              d->child = $1;
              if ($3 != nullptr) {
                  d->functionParams = *$3;
                  // parameter_type_list returns vector of Declarations that is not needed anymore.
                  delete $3;
              }
              $$ = d;
          }
          else
              $$ = nullptr;
      }
    | direct_declarator '(' ')'
      {
          if ($1 != nullptr) {
              Declarator *d = new Declarator(DK_FUNCTION, Union(@1, @3));
              d->child = $1;
              $$ = d;
          }
          else
              $$ = nullptr;
      }
    | direct_declarator '(' error ')'
    {
        $$ = nullptr;
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
        if ($1 != nullptr)
            dl->push_back($1);
        $$ = dl;
    }
    | parameter_list ',' parameter_declaration
    {
        std::vector<Declaration *> *dl = (std::vector<Declaration *> *)$1;
        if (dl == nullptr)
            dl = new std::vector<Declaration *>;
        if ($3 != nullptr)
            dl->push_back($3);
        $$ = dl;
    }
    | error ','
    {
        lSuggestParamListAlternates();
        $$ = nullptr;
    }
    ;

parameter_declaration
    : declaration_specifiers declarator
    {
        $$ = new Declaration($1, $2);
    }
    | declaration_specifiers declarator '=' initializer
    {
        if ($1 != nullptr && $2 != nullptr) {
            $2->initExpr = $4;
            $$ = new Declaration($1, $2);
        }
        else
            $$ = nullptr;
    }
    | declaration_specifiers abstract_declarator
    {
        if ($1 != nullptr && $2 != nullptr)
            $$ = new Declaration($1, $2);
        else
            $$ = nullptr;
    }
    | declaration_specifiers
    {
        if ($1 == nullptr)
            $$ = nullptr;
        else
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
        if ($1 == nullptr || $2 == nullptr)
            $$ = nullptr;
        else {
            $2->InitFromType($1, nullptr);
            $$ = $2->type;
        }
    }
    ;

abstract_declarator
    : pointer
      {
          $$ = $1;
      }
    | direct_abstract_declarator
    | pointer direct_abstract_declarator
      {
          if ($2 == nullptr)
              $$ = nullptr;
          else {
              Declarator *d = new Declarator(DK_POINTER, Union(@1, @2));
              d->child = $2;
              $$ = d;
          }
      }
    | reference
      {
          $$ = new Declarator(DK_REFERENCE, @1);
      }
    | reference direct_abstract_declarator
      {
          if ($2 == nullptr)
              $$ = nullptr;
          else {
              Declarator *d = new Declarator(DK_REFERENCE, Union(@1, @2));
              d->child = $2;
              $$ = d;
          }
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
        if ($2 != nullptr && lGetConstantInt($2, &size, @2, "Array dimension")) {
            if (size < 0) {
                Error(@2, "Array dimension must be non-negative.");
                $$ = nullptr;
            }
            else {
                Declarator *d = new Declarator(DK_ARRAY, Union(@1, @3));
                d->arraySize = size;
                $$ = d;
            }
        }
        else
            $$ = nullptr;
      }
    | direct_abstract_declarator '[' ']'
      {
          if ($1 == nullptr)
              $$ = nullptr;
          else {
              Declarator *d = new Declarator(DK_ARRAY, Union(@1, @3));
              d->arraySize = 0;
              d->child = $1;
              $$ = d;
          }
      }
    | direct_abstract_declarator '[' constant_expression ']'
      {
          int size;
          if ($1 != nullptr && $3 != nullptr && lGetConstantInt($3, &size, @3, "Array dimension")) {
              if (size < 0) {
                  Error(@3, "Array dimension must be non-negative.");
                  $$ = nullptr;
              }
              else {
                  Declarator *d = new Declarator(DK_ARRAY, Union(@1, @4));
                  d->arraySize = size;
                  d->child = $1;
                  $$ = d;
              }
          }
          else
              $$ = nullptr;
      }
    | '(' ')'
      { $$ = new Declarator(DK_FUNCTION, Union(@1, @2)); }
    | '(' parameter_type_list ')'
      {
          Declarator *d = new Declarator(DK_FUNCTION, Union(@1, @3));
          if ($2 != nullptr) {
              d->functionParams = *$2;
              // parameter_type_list returns vector of Declarations that is not needed anymore.
              delete $2;
          }
          $$ = d;
      }
    | direct_abstract_declarator '(' ')'
      {
          if ($1 == nullptr)
              $$ = nullptr;
          else {
              Declarator *d = new Declarator(DK_FUNCTION, Union(@1, @3));
              d->child = $1;
              $$ = d;
          }
      }
    | direct_abstract_declarator '(' parameter_type_list ')'
      {
          if ($1 == nullptr)
              $$ = nullptr;
          else {
              Declarator *d = new Declarator(DK_FUNCTION, Union(@1, @4));
              d->child = $1;
              if ($3 != nullptr) {
                  d->functionParams = *$3;
                  // parameter_type_list returns vector of Declarations that is not needed anymore.
                  delete $3;
              }
              $$ = d;
          }
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
          ExprList *exprList = $1;
          if (exprList == nullptr) {
              AssertPos(@1, m->errorCount > 0);
              exprList = new ExprList(@3);
          }
          exprList->exprs.push_back($3);
          exprList->pos = Union(exprList->pos, @3);
          $$ = exprList;
      }
    ;

pragma
    : TOKEN_PRAGMA
    {
        $$ = (yylval.pragmaAttributes);
    }
    ;

attributed_statement
    : pragma attributed_statement
    {
        if (($1->aType == PragmaAttributes::AttributeType::pragmaloop) && ($2 != nullptr)) {
            std::pair<Globals::pragmaUnrollType, int> unrollVal = std::pair<Globals::pragmaUnrollType, int>($1->unrollType, $1->count);
            $2->SetLoopAttribute(unrollVal);
        }
        $$ = $2;
        // deallocate yylval.pragmaAttributes returned from pragma and allocated in lPragmaUnroll
        delete $1;
    }
    | statement
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
    | delete_statement
    | unmasked_statement
    | error ';'
    {
        lSuggestBuiltinAlternates();
        $$ = nullptr;
    }
    ;

labeled_statement
    : goto_identifier ':' attributed_statement
    {
        $$ = new LabeledStmt($1, $3, @1);
        // allocated by strdup in goto_identifier
        free((char*)$1);
    }
    | TOKEN_CASE constant_expression ':' attributed_statement
      {
          int value;
          if ($2 != nullptr &&
              lGetConstantInt($2, &value, @2, "Case statement value")) {
              $$ = new CaseStmt(value, $4, Union(@1, @2));
          }
          else
              $$ = nullptr;
      }
    | TOKEN_DEFAULT ':' attributed_statement
      { $$ = new DefaultStmt($3, @1); }
    ;

start_scope
    : '{' { m->symbolTable->PushScope(); }
    ;

end_scope
    : '}' { m->symbolTable->PopScope(); }
    ;

compound_statement
    : '{' '}' { $$ = nullptr; }
    | start_scope statement_list end_scope { $$ = $2; }
    ;

statement_list
    : attributed_statement
      {
          StmtList *sl = new StmtList(@1);
          sl->Add($1);
          $$ = sl;
      }
    | statement_list attributed_statement
      {
          StmtList *sl = (StmtList *)$1;
          if (sl == nullptr) {
              AssertPos(@1, m->errorCount > 0);
              sl = new StmtList(@2);
          }
          sl->Add($2);
          $$ = sl;
      }
    ;

expression_statement
    : ';' { $$ = nullptr; }
    | expression ';' { $$ = $1 ? new ExprStmt($1, @1) : nullptr; }
    ;

selection_statement
    : TOKEN_IF '(' expression ')' attributed_statement
      { $$ = new IfStmt($3, $5, nullptr, false, @1); }
    | TOKEN_IF '(' expression ')' attributed_statement TOKEN_ELSE attributed_statement
      { $$ = new IfStmt($3, $5, $7, false, @1); }
    | TOKEN_CIF '(' expression ')' attributed_statement
      { $$ = new IfStmt($3, $5, nullptr, true, @1); }
    | TOKEN_CIF '(' expression ')' attributed_statement TOKEN_ELSE attributed_statement
      { $$ = new IfStmt($3, $5, $7, true, @1); }
    | TOKEN_SWITCH '(' expression ')' attributed_statement
      { $$ = new SwitchStmt($3, $5, @1); }
    ;

for_test
    : ';'
      { $$ = nullptr; }
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

foreach_scope
    : TOKEN_FOREACH { m->symbolTable->PushScope(); }
    ;

foreach_tiled_scope
    : TOKEN_FOREACH_TILED { m->symbolTable->PushScope(); }
    ;

foreach_identifier
    : TOKEN_IDENTIFIER
    {
        $$ = new Symbol(yytext, @1, AtomicType::VaryingInt32->GetAsConstType());
        lCleanUpString($1);
    }
    ;

foreach_active_scope
    : TOKEN_FOREACH_ACTIVE { m->symbolTable->PushScope(); }
    ;

foreach_active_identifier
    : TOKEN_IDENTIFIER
    {
        $$ = new Symbol(yytext, @1, AtomicType::UniformInt64->GetAsConstType());
        lCleanUpString($1);
    }
    ;

integer_dotdotdot
    : TOKEN_INT32DOTDOTDOT_CONSTANT {
        $$ = new ConstExpr(AtomicType::UniformInt32->GetAsConstType(),
                           (int32_t)yylval.intVal, @1);
    }
    | TOKEN_UINT32DOTDOTDOT_CONSTANT {
        $$ = new ConstExpr(AtomicType::UniformUInt32->GetAsConstType(),
                           (uint32_t)yylval.intVal, @1);
    }
    | TOKEN_INT64DOTDOTDOT_CONSTANT {
        $$ = new ConstExpr(AtomicType::UniformInt64->GetAsConstType(),
                           (int64_t)yylval.intVal, @1);
    }
    | TOKEN_UINT64DOTDOTDOT_CONSTANT {
        $$ = new ConstExpr(AtomicType::UniformUInt64->GetAsConstType(),
                           (uint64_t)yylval.intVal, @1);
    }
    ;

foreach_dimension_specifier
    : foreach_identifier '=' assignment_expression TOKEN_DOTDOTDOT assignment_expression
    {
        $$ = new ForeachDimension($1, $3, $5);
    }
    | foreach_identifier '=' integer_dotdotdot assignment_expression
    {
        $$ = new ForeachDimension($1, $3, $4);
    }
    ;

foreach_dimension_list
    : foreach_dimension_specifier
    {
        $$ = new std::vector<ForeachDimension *>;
        $$->push_back($1);
    }
    | foreach_dimension_list ',' foreach_dimension_specifier
    {
        std::vector<ForeachDimension *> *dv = $1;
        if (dv == nullptr) {
            AssertPos(@1, m->errorCount > 0);
            dv = new std::vector<ForeachDimension *>;
        }
        if ($3 != nullptr)
            dv->push_back($3);
        $$ = dv;
    }
    ;

foreach_unique_scope
    : TOKEN_FOREACH_UNIQUE { m->symbolTable->PushScope(); }
    ;

foreach_unique_identifier
    : TOKEN_IDENTIFIER
      {
          $$ = strdup($1->c_str());
          lCleanUpString($1);
      }
    ;

iteration_statement
    : TOKEN_WHILE '(' expression ')' attributed_statement
      { $$ = new ForStmt(nullptr, $3, nullptr, $5, false, @1); }
    | TOKEN_CWHILE '(' expression ')' attributed_statement
      { $$ = new ForStmt(nullptr, $3, nullptr, $5, true, @1); }
    | TOKEN_DO attributed_statement TOKEN_WHILE '(' expression ')' ';'
      { $$ = new DoStmt($5, $2, false, @1); }
    | TOKEN_CDO attributed_statement TOKEN_WHILE '(' expression ')' ';'
      { $$ = new DoStmt($5, $2, true, @1); }
    | for_scope '(' for_init_statement for_test ')' attributed_statement
      { $$ = new ForStmt($3, $4, nullptr, $6, false, @1);
        m->symbolTable->PopScope();
      }
    | for_scope '(' for_init_statement for_test expression ')' attributed_statement
      { $$ = new ForStmt($3, $4, new ExprStmt($5, @5), $7, false, @1);
        m->symbolTable->PopScope();
      }
    | cfor_scope '(' for_init_statement for_test ')' attributed_statement
      { $$ = new ForStmt($3, $4, nullptr, $6, true, @1);
        m->symbolTable->PopScope();
      }
    | cfor_scope '(' for_init_statement for_test expression ')' attributed_statement
      { $$ = new ForStmt($3, $4, new ExprStmt($5, @5), $7, true, @1);
        m->symbolTable->PopScope();
      }
    | foreach_scope '(' foreach_dimension_list ')'
     {
         std::vector<ForeachDimension *> *dims = $3;
         if (dims == nullptr) {
             AssertPos(@3, m->errorCount > 0);
             dims = new std::vector<ForeachDimension *>;
         }
         for (unsigned int i = 0; i < dims->size(); ++i)
             m->symbolTable->AddVariable((*dims)[i]->sym);
     }
     attributed_statement
     {
         std::vector<ForeachDimension *> *dims = $3;
         if (dims == nullptr) {
             AssertPos(@3, m->errorCount > 0);
             dims = new std::vector<ForeachDimension *>;
         }

         std::vector<Symbol *> syms;
         std::vector<Expr *> begins, ends;
         for (unsigned int i = 0; i < dims->size(); ++i) {
             syms.push_back((*dims)[i]->sym);
             begins.push_back((*dims)[i]->beginExpr);
             ends.push_back((*dims)[i]->endExpr);
         }
         $$ = new ForeachStmt(syms, begins, ends, $6, false, @1);
         m->symbolTable->PopScope();

         // deallocate ForeachDimension elements allocated in foreach_dimension_specifier
         for (unsigned int i = 0; i < dims->size(); ++i)
             delete (*dims)[i];

         // deallocate std::vector<ForeachDimension*> allocated in foreach_dimension_list
         delete dims;
     }
    | foreach_tiled_scope '(' foreach_dimension_list ')'
     {
         std::vector<ForeachDimension *> *dims = $3;
         if (dims == nullptr) {
             AssertPos(@3, m->errorCount > 0);
             dims = new std::vector<ForeachDimension *>;
         }

         for (unsigned int i = 0; i < dims->size(); ++i)
             m->symbolTable->AddVariable((*dims)[i]->sym);
     }
     attributed_statement
     {
         std::vector<ForeachDimension *> *dims = $3;
         if (dims == nullptr) {
             AssertPos(@1, m->errorCount > 0);
             dims = new std::vector<ForeachDimension *>;
         }

         std::vector<Symbol *> syms;
         std::vector<Expr *> begins, ends;
         for (unsigned int i = 0; i < dims->size(); ++i) {
             syms.push_back((*dims)[i]->sym);
             begins.push_back((*dims)[i]->beginExpr);
             ends.push_back((*dims)[i]->endExpr);
         }
         $$ = new ForeachStmt(syms, begins, ends, $6, true, @1);
         m->symbolTable->PopScope();

         // deallocate ForeachDimension elements allocated in foreach_dimension_specifier
         for (unsigned int i = 0; i < dims->size(); ++i)
             delete (*dims)[i];

         // deallocate std::vector<ForeachDimension*> allocated in foreach_dimension_list
         delete dims;
     }
    | foreach_active_scope '(' foreach_active_identifier ')'
     {
         if ($3 != nullptr)
             m->symbolTable->AddVariable($3);
     }
     attributed_statement
     {
         $$ = new ForeachActiveStmt($3, $6, Union(@1, @4));
         m->symbolTable->PopScope();
     }
    | foreach_unique_scope '(' foreach_unique_identifier TOKEN_IN
         expression ')'
     {
         Expr *expr = $5;
         const Type *type;
         if (expr != nullptr &&
             (expr = TypeCheck(expr)) != nullptr &&
             (type = expr->GetType()) != nullptr) {
             const Type *iterType = type->GetAsUniformType()->GetAsConstType();
             Symbol *sym = new Symbol($3, @3, iterType);
             m->symbolTable->AddVariable(sym);
         }
     }
     attributed_statement
     {
         $$ = new ForeachUniqueStmt($3, $5, $8, @1);
         m->symbolTable->PopScope();

         // allocated by strdup in foreach_unique_identifier
         free((char*)$3);
     }
    ;

goto_identifier
    : TOKEN_IDENTIFIER
      {
          $$ = strdup($1->c_str());
          lCleanUpString($1);
      }
    ;

jump_statement
    : TOKEN_GOTO goto_identifier ';'
      {
          $$ = new GotoStmt($2, @1, @2);
          // allocated by strdup in goto_identifier
          free((char*)$2);
      }
    | TOKEN_CONTINUE ';'
      { $$ = new ContinueStmt(@1); }
    | TOKEN_BREAK ';'
      { $$ = new BreakStmt(@1); }
    | TOKEN_RETURN ';'
      { $$ = new ReturnStmt(nullptr, @1); }
    | TOKEN_RETURN expression ';'
      { $$ = new ReturnStmt($2, @1); }
    ;

sync_statement
    : TOKEN_SYNC ';'
      { $$ = new ExprStmt(new SyncExpr(@1), @1); }
    ;

delete_statement
    : TOKEN_DELETE expression ';'
    {
        $$ = new DeleteStmt($2, Union(@1, @2));
    }
    ;

unmasked_statement
    : TOKEN_UNMASKED '{' statement_list '}'
    {
        $$ = new UnmaskedStmt($3, @1);
    }
    ;

print_statement
    : TOKEN_PRINT '(' string_constant ')' ';'
      {
           $$ = new PrintStmt(*$3, nullptr, @1);
           // deallocate std::string of string_constant
           lCleanUpString($3);
      }
    | TOKEN_PRINT '(' string_constant ',' argument_expression_list ')' ';'
      {
           $$ = new PrintStmt(*$3, $5, @1);
           // deallocate std::string of string_constant
           lCleanUpString($3);
      }
    ;

assert_statement
    : TOKEN_ASSERT '(' string_constant ',' expression ')' ';'
      {
          $$ = new AssertStmt(*$3, $5, @1);
          // deallocate std::string of string_constant
          lCleanUpString($3);
      }
    ;

translation_unit
    : external_declaration
    | translation_unit external_declaration
    | error ';'
    ;

external_declaration
    : function_definition
    | template_function_declaration_or_definition
    | template_function_specialization
    | template_function_instantiation
    | TOKEN_EXTERN TOKEN_STRING_C_LITERAL '{' declaration '}'
    | TOKEN_EXTERN TOKEN_STRING_SYCL_LITERAL '{' declaration '}'
    | TOKEN_EXPORT '{' type_specifier_list '}' ';'
    {
        if ($3 != nullptr)
            m->AddExportedTypes(*$3);
    }
    | declaration
    {
        if ($1 != nullptr)
            for (unsigned int i = 0; i < $1->declarators.size(); ++i)
                lAddDeclaration($1->declSpecs, $1->declarators[i]);
    }
    | ';'
    ;

function_definition
    : declaration_specifiers declarator
    {
        lAddDeclaration($1, $2);
        m->symbolTable->PushScope();
        lAddFunctionParams($2);
        lAddMaskToSymbolTable(@2);
        if ($1->typeQualifiers & TYPEQUAL_TASK)
            lAddThreadIndexCountToSymbolTable(@2);
    }
    compound_statement
    {
        if ($2 != nullptr) {
            // FIXME: Next list is redundant, as it's done in lAddDeclaration()
            $2->InitFromDeclSpecs($1);
            const FunctionType *funcType = CastType<FunctionType>($2->type);
            if (funcType == nullptr)
                AssertPos(@1, m->errorCount > 0);
            else if ($1->storageClass == SC_TYPEDEF)
                Error(@1, "Illegal \"typedef\" provided with function definition.");
            else {
                Stmt *code = $4;
                if (code == nullptr) code = new StmtList(@4);
                m->AddFunctionDefinition($2->name, funcType, code);
            }
        }
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

template_type_parameter
    : TOKEN_TYPENAME TOKEN_IDENTIFIER
      {
          $$ = new TemplateTypeParmType(*$<stringVal>2, Variability::VarType::Unbound, false, Union(@1, @2));
          lCleanUpString($<stringVal>2);
      }
    | TOKEN_TYPENAME TOKEN_IDENTIFIER '=' type_specifier
      {
          $$ = new TemplateTypeParmType(*$<stringVal>2, Variability::VarType::Unbound, false, Union(@1, @2));
          lCleanUpString($<stringVal>2);
          // TODO: implement
          Error(@4, "Default values for template type parameters are not yet supported.");
      }
    ;

int_constant_type
    : TOKEN_INT8  { $$ = AtomicType::UniformInt8->GetAsConstType(); }
    | TOKEN_INT16 { $$ = AtomicType::UniformInt16->GetAsConstType(); }
    | TOKEN_INT   { $$ = AtomicType::UniformInt32->GetAsConstType(); }
    | TOKEN_INT64 { $$ = AtomicType::UniformInt64->GetAsConstType(); }
    | TOKEN_UINT8 { $$ = AtomicType::UniformUInt8->GetAsConstType(); }
    | TOKEN_UINT16{ $$ = AtomicType::UniformUInt16->GetAsConstType(); }
    | TOKEN_UINT  { $$ = AtomicType::UniformUInt32->GetAsConstType(); }
    | TOKEN_UINT64{ $$ = AtomicType::UniformUInt64->GetAsConstType(); }
    ;

template_int_constant_type
    : TOKEN_UNIFORM int_constant_type { $$ = $2; }
    | int_constant_type { $$ = $1;}
    ;

template_int_parameter
    : template_int_constant_type TOKEN_IDENTIFIER
      {
          $$ = new Symbol(*$<stringVal>2, Union(@1, @2), $1);
          lCleanUpString($2);
      }
      | template_int_constant_type TOKEN_IDENTIFIER '=' int_constant
      {
          $$ = new Symbol(*$<stringVal>2, Union(@1, @2), $1);
          lCleanUpString($2);
          // TODO: implement
          Error(@4, "Default values for template non-type parameters are not yet supported.");
      }
    ;

template_enum_parameter
    : TOKEN_TYPE_NAME TOKEN_IDENTIFIER
      {
          const Type *type = m->symbolTable->LookupType($1->c_str());
          const EnumType *enumType = CastType<EnumType>(type);
          if (enumType == nullptr) {
            Error(@1, "Only enum types and integral types are allowed as non-type template parameters.");
          }
          $$ = new Symbol(*$<stringVal>2, Union(@1, @2), enumType->GetAsConstType()->GetAsUniformType());
          lCleanUpString($1);
          lCleanUpString($2);
      }

template_parameter
    : template_type_parameter
    {
        if ($1 != nullptr) {
            $$ = new TemplateParam($1);
        }
    }
    | template_int_parameter
    {
        if ($1 != nullptr) {
            $$ = new TemplateParam($1);
        }
    }
    | template_enum_parameter
    {
        if ($1 != nullptr) {
            $$ = new TemplateParam($1);
        }
    }
    ;

template_parameter_list
    : template_parameter
      {
          TemplateParms *list = new TemplateParms();
          if ($1 != nullptr) {
              list->Add($1);
          }
          $$ = list;
      }
    | template_parameter_list ',' template_parameter
      {
          TemplateParms *list = (TemplateParms *) $1;
          if (list == nullptr) {
              AssertPos(@1, m->errorCount > 0);
              list = new TemplateParms();
          }
          if ($3 != nullptr) {
              list->Add($3);
          }
          $$ = list;
      }
    ;

template_head
    : TOKEN_TEMPLATE '<' template_parameter_list '>'
      {
          $$ = $3;
      }
    ;

template_declaration
    : template_head
      {
          // Scope for template parameters definition
          m->symbolTable->PushScope();
          TemplateParms *list = (TemplateParms *) $1;
          for(size_t i = 0; i < list->GetCount(); i++) {
              std::string name = (*list)[i]->GetName();
              SourcePos pos = (*list)[i]->GetSourcePos();
              if ((*list)[i]->IsTypeParam()) {
                  m->AddTypeDef(name, (*list)[i]->GetTypeParam(), pos);
              } else if ((*list)[i]->IsNonTypeParam()) {
                  m->symbolTable->AddVariable((*list)[i]->GetNonTypeParam());
              }
          }
      }
      declaration_specifiers declarator
      {
          lAddTemplateDeclaration($1, $3, $4);
          lAddFunctionParams($4);
          lAddMaskToSymbolTable(@4);
          const FunctionType *ft = CastType<FunctionType>($4->type);
          // Creating a new TemplateSymbol just to pass it further seems to be a waste
          $$ = new TemplateSymbol($1, $4->name, ft, $3->storageClass, @4, false /*not used*/, false /*not used*/);
      }
    ;

template_function_declaration_or_definition
    : template_declaration ';'
      {
          // deallocate TemplateSymbol created in template_declaration
          delete $1;
          // End templates parameters definition scope
          m->symbolTable->PopScope();
      }
    | template_declaration compound_statement
      {
          if ($1 != nullptr) {
              Stmt *code = $2;
              if (code == nullptr) code = new StmtList(@2);
              m->AddFunctionTemplateDefinition($1->templateParms, $1->name, $1->type, code);
              // deallocate TemplateSymbol created in template_declaration
              delete $1;
          }

          // End templates parameters definition scope
          m->symbolTable->PopScope();
      }
    ;

template_argument
    : rate_qualified_type_specifier
    {
        $$ = new TemplateArg($1, @1);
    }
    // Ideally we should use here constant_expression, however, there is grammar ambiguitiy between
    // template_identifier '<' template_argument_list '>' in simple_template_id and
    // relational_expression '<' shift_expression in relational_expression (part of constant_expression).
    | TOKEN_INT8_CONSTANT {
        $$ = new TemplateArg(new ConstExpr(AtomicType::UniformInt8->GetAsConstType(),
                           (int8_t)yylval.intVal, @1), @1);
    }
    | TOKEN_UINT8_CONSTANT {
        $$ = new TemplateArg(new ConstExpr(AtomicType::UniformUInt8->GetAsConstType(),
                           (uint8_t)yylval.intVal, @1), @1);
    }
    | TOKEN_INT16_CONSTANT {
        $$ = new TemplateArg(new ConstExpr(AtomicType::UniformInt16->GetAsConstType(),
                           (int16_t)yylval.intVal, @1), @1);
    }
    | TOKEN_UINT16_CONSTANT {
        $$ = new TemplateArg(new ConstExpr(AtomicType::UniformUInt16->GetAsConstType(),
                           (uint16_t)yylval.intVal, @1), @1);
    }
    | TOKEN_INT32_CONSTANT {
        $$ = new TemplateArg(new ConstExpr(AtomicType::UniformInt32->GetAsConstType(),
                           (int32_t)yylval.intVal, @1), @1);
    }
    | TOKEN_UINT32_CONSTANT {
        $$ = new TemplateArg(new ConstExpr(AtomicType::UniformUInt32->GetAsConstType(),
                           (uint32_t)yylval.intVal, @1), @1);
    }
    | TOKEN_INT64_CONSTANT {
        $$ = new TemplateArg(new ConstExpr(AtomicType::UniformInt64->GetAsConstType(),
                           (int64_t)yylval.intVal, @1), @1);
    }
    | TOKEN_UINT64_CONSTANT {
        $$ = new TemplateArg(new ConstExpr(AtomicType::UniformUInt64->GetAsConstType(),
                           (uint64_t)yylval.intVal, @1), @1);
    }
    // Enums and nested templates case:
    | TOKEN_IDENTIFIER
    {
        const char *name = $1->c_str();
        Symbol *s = m->symbolTable->LookupVariable(name);
        if (s) {
            $$ = new TemplateArg(new SymbolExpr(s, @1), @1);
        } else {
            Error(@1, "Unknown identifier");
            $$ = nullptr;
        }
        lCleanUpString($1);
    }
    ;

template_argument_list
    : template_argument
      {
          TemplateArgs *templArgs = new TemplateArgs();
          if ($1 != nullptr) {
            templArgs->push_back(*$1);
          }
          $$ = templArgs;

      }
    | template_argument_list ',' template_argument
      {
          TemplateArgs *templArgs = (TemplateArgs *) $1;
          if ($3 != nullptr) {
            templArgs->push_back(*$3);
          }
          $$ = templArgs;
      }
    ;

template_identifier
    : TOKEN_TEMPLATE_NAME
    {
        $$ = strdup(yytext);
        lCleanUpString($1);
    }
    ;

simple_template_id
    : template_identifier '<' template_argument_list '>'
      {
          // Template ID declartor
          Declarator *d = new Declarator(DK_BASE, @1);
          d->name = $1;
          // allocated by strdup in template_identifier
          free((char*)$1);
          // Arguments vector
          TemplateArgs *templArgs = (TemplateArgs *) $3;
          // Bundle template ID declarator and type list.
          $$ = new std::pair(d, templArgs);
      }
    | template_identifier
      {
          // Template ID declartor
          Declarator *d = new Declarator(DK_BASE, @1);
          d->name = $1;
          // allocated by strdup in template_identifier
          free((char*)$1);
          // Arguments vector
          TemplateArgs *templArgs = new TemplateArgs();
          // Bundle template ID declarator and empty type list.
          $$ = new std::pair(d, templArgs);
      }

    ;

 // template int foo<int>(int);
template_function_instantiation
    : TOKEN_TEMPLATE declaration_specifiers simple_template_id '(' parameter_type_list ')' ';'
      {
          SimpleTemplateIDType *simpleTemplID = (SimpleTemplateIDType *) $3;

          // Function declarator
          Declarator *d = new Declarator(DK_FUNCTION, Union(@1, @6));
          d->child = simpleTemplID->first;
          if ($5 != nullptr) {
              d->functionParams = *$5;
              // parameter_type_list returns vector of Declarations that is not needed anymore.
              delete $5;
          }

          d->InitFromDeclSpecs($2);
          lCheckTemplateDeclSpecs($2, d->pos, TemplateType::Instantiation, $3->first->name.c_str());
          const FunctionType *ftype = CastType<FunctionType>(d->type);
          bool isInline = ($2->typeQualifiers & TYPEQUAL_INLINE);
          bool isNoInline = ($2->typeQualifiers & TYPEQUAL_NOINLINE);
          if ($3->second->size() == 0) {
              Error(d->pos, "Template arguments deduction is not yet supported in explicit template instantiation.");
          }
          m->AddFunctionTemplateInstantiation($3->first->name, *$3->second, ftype, $2->storageClass, isInline, isNoInline, Union(@1, @6));

          // deallocate SimpleTemplateIDType returned by simple_template_id
          lFreeSimpleTemplateID(simpleTemplID);
      }
    | TOKEN_TEMPLATE declaration_specifiers simple_template_id '(' ')' ';'
      {
          SimpleTemplateIDType *simpleTemplID = (SimpleTemplateIDType *) $3;

          // Function declarator
          Declarator *d = new Declarator(DK_FUNCTION, Union(@1, @5));
          d->child = simpleTemplID->first;

          d->InitFromDeclSpecs($2);
          lCheckTemplateDeclSpecs($2, d->pos, TemplateType::Instantiation, $3->first->name.c_str());
          const FunctionType *ftype = CastType<FunctionType>(d->type);
          bool isInline = ($2->typeQualifiers & TYPEQUAL_INLINE);
          bool isNoInline = ($2->typeQualifiers & TYPEQUAL_NOINLINE);
          if ($3->second->size() == 0) {
              Error(d->pos, "Template arguments deduction is not yet supported in explicit template instantiation.");
          }
          m->AddFunctionTemplateInstantiation($3->first->name, *$3->second, ftype, $2->storageClass, isInline, isNoInline, Union(@1, @5));

          // deallocate SimpleTemplateIDType returned by simple_template_id
          lFreeSimpleTemplateID(simpleTemplID);
      }
    | TOKEN_TEMPLATE declaration_specifiers simple_template_id '(' error ')' ';'
      {
          // deallocate SimpleTemplateIDType returned by simple_template_id
          lFreeSimpleTemplateID($3);
      }
    ;

// Template specialization, a-la
// template <> int foo<int>(int) { ... }
template_function_specialization_declaration
    : TOKEN_TEMPLATE '<' '>' declaration_specifiers simple_template_id '(' parameter_type_list ')'
      {
        // Function declarator
        Declarator *d = new Declarator(DK_FUNCTION, Union(@1, @8));
        d->child = $5->first;

        if ($7 != nullptr) {
            d->functionParams = *$7;
            // parameter_type_list returns vector of Declarations that is not needed anymore.
            delete $7;
        }
        TemplateArgs *templArgs = new TemplateArgs(*$5->second);
        Assert(templArgs);
        lAddTemplateSpecialization(*templArgs, $4, d);
        m->symbolTable->PushScope();

        lAddFunctionParams(d);
        lAddMaskToSymbolTable(@5);
        // deallocate SimpleTemplateIDType returned by simple_template_id
        lFreeSimpleTemplateID($5);
        $$ = new std::pair(d, templArgs);
      }
    | TOKEN_TEMPLATE '<' '>' declaration_specifiers simple_template_id '(' ')'
      {
        Declarator *d = new Declarator(DK_FUNCTION, Union(@1, @5));
        d->child = $5->first;
        TemplateArgs *templArgs = new TemplateArgs(*$5->second);
        Assert(templArgs);
        lAddTemplateSpecialization(*templArgs, $4, d);
        m->symbolTable->PushScope();

        lAddMaskToSymbolTable(@5);
        // deallocate SimpleTemplateIDType returned by simple_template_id
        lFreeSimpleTemplateID($5);
        $$ = new std::pair(d, templArgs);
      }
    | TOKEN_TEMPLATE '<' '>' declaration_specifiers simple_template_id '(' error ')'
      {
        m->symbolTable->PushScope();
        // deallocate SimpleTemplateIDType returned by simple_template_id
        lFreeSimpleTemplateID($5);
        $$ = nullptr;
      }
    ;

template_function_specialization
    : template_function_specialization_declaration ';'
      {
          if ($1 != nullptr) {
            // deallocate TemplateSymbol created in template_declaration
            lFreeSimpleTemplateID($1);
          }
          // End templates parameters definition scope
          m->symbolTable->PopScope();
      }
    | template_function_specialization_declaration compound_statement
      {
        if ($1 != nullptr) {
            Declarator *d = $1->first;
            const FunctionType *ftype = CastType<FunctionType>(d->type);
            if (ftype == nullptr)
                AssertPos(@1, m->errorCount > 0);
            else {
                Stmt *code = $2;
                if (code == nullptr) code = new StmtList(@2);
                m->AddFunctionTemplateSpecializationDefinition(d->name, ftype, *$1->second, Union(@1, @2), code);
            }
           lFreeSimpleTemplateID($1);
        }
        m->symbolTable->PopScope();
      }
    ;

%%


void yyerror(const char *s) {
    if (strlen(yytext) == 0)
        Error(yylloc, "Premature end of file: %s.", s);
    else
        Error(yylloc, "%s.", s);
}

void lCleanUpString(std::string *s) {
    if (s) {
         delete s;
    }
}

void lFreeSimpleTemplateID(void *p) {
    SimpleTemplateIDType *sid = (SimpleTemplateIDType*) p;
    TemplateArgs *templArgs = sid->second;
    if (templArgs) {
        delete templArgs;
    }
    if (sid) {
        delete sid;
    }
}

static int
lYYTNameErr (char *yyres, const char *yystr)
{
  extern std::map<std::string, std::string> tokenNameRemap;
  Assert(tokenNameRemap.size() > 0);
  if (tokenNameRemap.find(yystr) != tokenNameRemap.end()) {
      std::string n = tokenNameRemap[yystr];
      if (yyres == nullptr)
          return n.size();
      else
          return yystpcpy(yyres, n.c_str()) - yyres;
  }

  if (*yystr == '"')
    {
      YYSIZE_T yyn = 0;
      char const *yyp = yystr;

      for (;;)
        switch (*++yyp)
          {
          case '\'':
          case ',':
            goto do_not_strip_quotes;

          case '\\':
            if (*++yyp != '\\')
              goto do_not_strip_quotes;
            /* Fall through.  */
          default:
            if (yyres)
              yyres[yyn] = *yyp;
            yyn++;
            break;

          case '"':
            if (yyres)
              yyres[yyn] = '\0';
            return yyn;
          }
    do_not_strip_quotes: ;
    }

  if (! yyres)
    return yystrlen (yystr);

  return yystpcpy (yyres, yystr) - yyres;
}

static void
lSuggestBuiltinAlternates() {
    std::vector<std::string> builtinTokens;
    const char **token = lBuiltinTokens;
    while (*token) {
        builtinTokens.push_back(*token);
        ++token;
    }
    std::vector<std::string> alternates = MatchStrings(yytext, builtinTokens);
    std::string alts = lGetAlternates(alternates);
    if (alts.size() > 0)
         Error(yylloc, "%s", alts.c_str());
}


static void
lSuggestParamListAlternates() {
    std::vector<std::string> builtinTokens;
    const char **token = lParamListTokens;
    while (*token) {
        builtinTokens.push_back(*token);
        ++token;
    }
    std::vector<std::string> alternates = MatchStrings(yytext, builtinTokens);
    std::string alts = lGetAlternates(alternates);
    if (alts.size() > 0)
        Error(yylloc, "%s", alts.c_str());
}


static void
lAddDeclaration(DeclSpecs *ds, Declarator *decl) {
    if (ds == nullptr || decl == nullptr)
        // Error happened earlier during parsing
        return;

    decl->InitFromDeclSpecs(ds);
    if (ds->storageClass == SC_TYPEDEF)
        m->AddTypeDef(decl->name, decl->type, decl->pos);
    else {
        if (decl->type == nullptr) {
            Assert(m->errorCount > 0);
            return;
        }

        decl->type = decl->type->ResolveUnboundVariability(Variability::Varying);

        const FunctionType *ft = CastType<FunctionType>(decl->type);
        if (ft != nullptr) {
            bool isInline = (ds->typeQualifiers & TYPEQUAL_INLINE);
            bool isNoInline = (ds->typeQualifiers & TYPEQUAL_NOINLINE);
            bool isVectorCall = (ds->typeQualifiers & TYPEQUAL_VECTORCALL);
            bool isRegCall = (ds->typeQualifiers & TYPEQUAL_REGCALL);
            m->AddFunctionDeclaration(decl->name, ft, ds->storageClass,
                                      isInline, isNoInline, isVectorCall, isRegCall, decl->pos);
        }
        else {
            bool isConst = (ds->typeQualifiers & TYPEQUAL_CONST) != 0;
            m->AddGlobalVariable(decl->name, decl->type, decl->initExpr,
                                 isConst, decl->storageClass, decl->pos);
        }
    }
}

static void
lCheckTemplateDeclSpecs(DeclSpecs *ds, SourcePos pos, TemplateType type, const char* name) {
    std::string templateTypeStr;
    switch (type) {
        case TemplateType::Template:
            templateTypeStr = "function template";
            break;
        case TemplateType::Instantiation:
            templateTypeStr = "template instantiation";
            break;
        case TemplateType::Specialization:
            templateTypeStr = "template specialization";
            break;
        default:
            FATAL("Unhandled template type in lCheckTemplateDeclSpecs");
    }
    if (ds->typeQualifiers & TYPEQUAL_TASK){
        Error(pos, "'task' not supported for %s.", templateTypeStr.c_str());
        return;
    }
    if (ds->typeQualifiers & TYPEQUAL_EXPORT) {
        Error(pos, "'export' not supported for %s.", templateTypeStr.c_str());
        return;
    }
    if (ds->storageClass == SC_TYPEDEF) {
        Error(pos, "Illegal \"typedef\" provided with %s.", templateTypeStr.c_str());
        return;
    }
    // We can't support extern "C"/extern "SYCL" for templates because
    // we need mangling information.
    if (ds->storageClass == SC_EXTERN_C || ds->storageClass == SC_EXTERN_SYCL) {
        Error(pos, "Illegal linkage provided with %s.", templateTypeStr.c_str());
        return;
    }
    Assert(ds->storageClass == SC_NONE || ds->storageClass == SC_STATIC || ds->storageClass == SC_EXTERN);
    bool isVectorCall = (ds->typeQualifiers & TYPEQUAL_VECTORCALL);
    if (isVectorCall) {
        Error(pos, "Illegal to use \"__vectorcall\" qualifier on non-extern function \"%s\".", name);
    }
    bool isRegCall = (ds->typeQualifiers & TYPEQUAL_REGCALL);
    if (isRegCall) {
        Error(pos, "Illegal to use \"__regcall\" qualifier on non-extern function \"%s\".", name);
    }
}

static void
lAddTemplateDeclaration(TemplateParms *templateParmList, DeclSpecs *ds, Declarator *decl) {
    if (ds == nullptr || decl == nullptr) {
        // Error happened earlier during parsing
        return;
    }

    decl->InitFromDeclSpecs(ds);
    lCheckTemplateDeclSpecs(ds, decl->pos, TemplateType::Template, decl->name.c_str());

    if (decl->type == nullptr) {
        Assert(m->errorCount > 0);
        return;
    }

    const FunctionType *ft = CastType<FunctionType>(decl->type);
    if (ft != nullptr) {
        bool isInline = (ds->typeQualifiers & TYPEQUAL_INLINE);
        bool isNoInline = (ds->typeQualifiers & TYPEQUAL_NOINLINE);
        m->AddFunctionTemplateDeclaration(templateParmList, decl->name, ft, ds->storageClass,
                                          isInline, isNoInline, decl->pos);
    }
    else {
        Error(decl->pos, "Only function templates are supported.");
    }

}

static void
lAddTemplateSpecialization(const TemplateArgs &templArgs, DeclSpecs *ds, Declarator *decl) {
    if (ds == nullptr || decl == nullptr)
        // Error happened earlier during parsing
        return;

    decl->InitFromDeclSpecs(ds);
    lCheckTemplateDeclSpecs(ds, decl->pos, TemplateType::Specialization, decl->name.c_str());

    if (decl->type == nullptr) {
        Assert(m->errorCount > 0);
        return;
    }

    if (templArgs.size() == 0) {
        Error(decl->pos, "Template arguments deduction is not yet supported in template function specialization.");
        return;
    }

    const FunctionType *ftype = CastType<FunctionType>(decl->type);
    if (ftype != nullptr) {
        bool isInline = (ds->typeQualifiers & TYPEQUAL_INLINE);
        bool isNoInline = (ds->typeQualifiers & TYPEQUAL_NOINLINE);
        m->AddFunctionTemplateSpecializationDeclaration(decl->name, ftype, templArgs, ds->storageClass,
                                                        isInline, isNoInline, decl->pos);
    }
    else {
        Error(decl->pos, "Only function template specializations are supported.");
    }
}

/** We're about to start parsing the body of a function; add all of the
    parameters to the symbol table so that they're available.
*/
static void
lAddFunctionParams(Declarator *decl) {
    // It's responsibility of the caller to create a new symbol table scope.
    // For regular functions, the scope starts before function parameters.
    // For template functions, the scope starts before template parameters.
    if (decl == nullptr) {
        return;
    }

    // walk down to the declarator for the function itself
    while (decl->kind != DK_FUNCTION && decl->child != nullptr)
        decl = decl->child;
    if (decl->kind != DK_FUNCTION) {
        AssertPos(decl->pos, m->errorCount > 0);
        return;
    }

    // now loop over its parameters and add them to the symbol table
    for (unsigned int i = 0; i < decl->functionParams.size(); ++i) {
        Declaration *pdecl = decl->functionParams[i];
        Assert(pdecl != nullptr && pdecl->declarators.size() == 1);
        Declarator *declarator = pdecl->declarators[0];
        if (declarator == nullptr)
            AssertPos(decl->pos, m->errorCount > 0);
        else {
            Symbol *sym = new Symbol(declarator->name, declarator->pos,
                                     declarator->type, declarator->storageClass);
#ifndef NDEBUG
            bool ok = m->symbolTable->AddVariable(sym);
            if (ok == false)
                AssertPos(decl->pos, m->errorCount > 0);
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
    const Type *t = nullptr;
    switch (g->target->getMaskBitCount()) {
    case 1:
        t = AtomicType::VaryingBool;
        break;
    case 8:
        t = AtomicType::VaryingUInt8;
        break;
    case 16:
        t = AtomicType::VaryingUInt16;
        break;
    case 32:
        t = AtomicType::VaryingUInt32;
        break;
    case 64:
        t = AtomicType::VaryingUInt64;
        break;
    default:
        FATAL("Unhandled mask bitsize in lAddMaskToSymbolTable");
    }

    t = t->GetAsConstType();
    Symbol *maskSymbol = new Symbol("__mask", pos, t);
    m->symbolTable->AddVariable(maskSymbol);
}


/** Add the thread index and thread count variables to the symbol table
    (this should only be done for 'task'-qualified functions. */
static void lAddThreadIndexCountToSymbolTable(SourcePos pos) {
    const Type *type = AtomicType::UniformUInt32->GetAsConstType();

    Symbol *threadIndexSym = new Symbol("threadIndex", pos, type);
    m->symbolTable->AddVariable(threadIndexSym);

    Symbol *threadCountSym = new Symbol("threadCount", pos, type);
    m->symbolTable->AddVariable(threadCountSym);

    Symbol *taskIndexSym = new Symbol("taskIndex", pos, type);
    m->symbolTable->AddVariable(taskIndexSym);

    Symbol *taskCountSym = new Symbol("taskCount", pos, type);
    m->symbolTable->AddVariable(taskCountSym);

    Symbol *taskIndexSym0 = new Symbol("taskIndex0", pos, type);
    m->symbolTable->AddVariable(taskIndexSym0);
    Symbol *taskIndexSym1 = new Symbol("taskIndex1", pos, type);
    m->symbolTable->AddVariable(taskIndexSym1);
    Symbol *taskIndexSym2 = new Symbol("taskIndex2", pos, type);
    m->symbolTable->AddVariable(taskIndexSym2);


    Symbol *taskCountSym0 = new Symbol("taskCount0", pos, type);
    m->symbolTable->AddVariable(taskCountSym0);
    Symbol *taskCountSym1 = new Symbol("taskCount1", pos, type);
    m->symbolTable->AddVariable(taskCountSym1);
    Symbol *taskCountSym2 = new Symbol("taskCount2", pos, type);
    m->symbolTable->AddVariable(taskCountSym2);
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
    case SC_STATIC:
        return "static";
    case SC_TYPEDEF:
        return "typedef";
    case SC_EXTERN_C:
        return "extern \"C\"";
    case SC_EXTERN_SYCL:
        return "extern \"SYCL\"";
    default:
        Assert(!"logic error in lGetStorageClassString()");
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
    if (expr == nullptr)
        return false;
    expr = TypeCheck(expr);
    if (expr == nullptr)
        return false;
    expr = Optimize(expr);
    if (expr == nullptr)
        return false;

    std::pair<llvm::Constant *, bool> cValPair = expr->GetConstant(expr->GetType());
    llvm::Constant *cval = cValPair.first;
    if (cval == nullptr) {
        Error(pos, "%s must be a compile-time constant.", usage);
        return false;
    }
    else {
        llvm::ConstantInt *ci = llvm::dyn_cast<llvm::ConstantInt>(cval);
        if (ci == nullptr) {
            Error(pos, "%s must be a compile-time integer constant.", usage);
            return false;
        }
        if ((int64_t)((int32_t)ci->getSExtValue()) != ci->getSExtValue()) {
            Error(pos, "%s must be representable with a 32-bit integer.", usage);
            return false;
        }
        const Type *type = expr->GetType();
        if (type->IsUnsignedType())
            *value = (int)ci->getZExtValue();
        else
            *value = (int)ci->getSExtValue();
        return true;
    }
}


static EnumType *
lCreateEnumType(const char *name, std::vector<Symbol *> *enums, SourcePos pos) {
    if (enums == nullptr)
        return nullptr;

    EnumType *enumType = name ? new EnumType(name, pos) : new EnumType(pos);
    if (name != nullptr)
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
        if (enums[i]->constValue != nullptr) {
            /* Already has a value, so first update nextVal with it. */
            int count = enums[i]->constValue->GetValues(&nextVal);
            AssertPos(enums[i]->pos, count == 1);
            ++nextVal;

            /* When the source file as being parsed, the ConstExpr for any
               enumerant with a specified value was set to have unsigned
               int32 type, since we haven't created the parent EnumType
               by then.  Therefore, add a little type cast from uint32 to
               the actual enum type here and optimize it, which will have
               us end up with a ConstExpr with the desired EnumType... */
            Expr *castExpr = new TypeCastExpr(enumType, enums[i]->constValue,
                                              enums[i]->pos);
            castExpr = Optimize(castExpr);
            enums[i]->constValue = llvm::dyn_cast<ConstExpr>(castExpr);
            AssertPos(enums[i]->pos, enums[i]->constValue != nullptr);
        }
        else {
            enums[i]->constValue = new ConstExpr(enumType, nextVal++,
                                                 enums[i]->pos);
        }
    }
}
