
	code for sm_35
		Function : mandelbrot_scanline
	.headerflags    @"EF_CUDA_SM35 EF_CUDA_PTX_SM(EF_CUDA_SM35)"
                                                                            /* 0x0880a010a0a01000 */
        /*0008*/                MOV R1, c[0x0][0x44];                       /* 0x64c03c00089c0006 */
        /*0010*/                S2R R4, SR_CTAID.Y;                         /* 0x86400000131c0012 */
        /*0018*/                MOV R6, c[0x0][0x158];                      /* 0x64c03c002b1c001a */
        /*0020*/                IMUL R0, R4, c[0x0][0x15c];                 /* 0x61c018002b9c1002 */
        /*0028*/                IADD R3, R0, c[0x0][0x15c];                 /* 0x608000002b9c000e */
        /*0030*/                S2R R2, SR_CTAID.X;                         /* 0x86400000129c000a */
        /*0038*/                IMNMX R5, R3, c[0x0][0x154], PT;            /* 0x61081c002a9c0c16 */
                                                                            /* 0x08a010a000b010a0 */
        /*0048*/                IMAD R3, R2, c[0x0][0x158], R6;             /* 0x510818002b1c080e */
        /*0050*/                ISETP.GE.AND P0, PT, R0, R5, PT;            /* 0xdb681c00029c001e */
        /*0058*/                IMNMX R3, R3, c[0x0][0x150], PT;            /* 0x61081c002a1c0c0e */
        /*0060*/            @P0 EXIT ;                                      /* 0x180000000000003c */
        /*0068*/                IADD R4, R4, 0x1;                           /* 0xc0800000009c1011 */
        /*0070*/                IMUL R5, R4, c[0x0][0x15c];                 /* 0x61c018002b9c1016 */
        /*0078*/                LOP.PASS_B R4, RZ, ~c[0x0][0x154];          /* 0x620038002a9ffc12 */
                                                                            /* 0x0800b0a01000a0a0 */
        /*0088*/                LOP.PASS_B R5, RZ, ~R5;                     /* 0xe2003800029ffc16 */
        /*0090*/                IMNMX R4, R4, R5, !PT;                      /* 0xe1083c00029c1012 */
        /*0098*/                LOP.PASS_B R4, RZ, ~R4;                     /* 0xe2003800021ffc12 */
        /*00a0*/                IMUL R5, R2, c[0x0][0x158];                 /* 0x61c018002b1c0816 */
        /*00a8*/                SSY 0x318;                                  /* 0x1480000134000000 */
        /*00b0*/                ISETP.GE.AND P0, PT, R5, R3, PT;            /* 0xdb681c00019c141e */
        /*00b8*/            @P0 BRA 0x310;                                  /* 0x120000012800003c */
                                                                            /* 0x08a0a00010ac8010 */
        /*00c8*/                ISETP.LT.AND P0, PT, RZ, c[0x0][0x160], PT; /* 0x5b181c002c1ffc1e */
        /*00d0*/                I2F.F32.S32 R6, R0;                         /* 0xe5c00000001ca81a */
        /*00d8*/                MOV R7, c[0x0][0x148];                      /* 0x64c03c00291c001e */
        /*00e0*/                FFMA R6, R6, c[0x0][0x14c], R7;             /* 0x4c001c00299c181a */
        /*00e8*/            @P0 BRA 0x180;                                  /* 0x120000004800003c */
        /*00f0*/                S2R R7, SR_TID.X;                           /* 0x86400000109c001e */
        /*00f8*/                LOP.AND R6, R7, 0x1f;                       /* 0xc20000000f9c1c19 */
                                                                            /* 0x08a010a0a080b0a0 */
        /*0108*/                IADD R6, R6, R5;                            /* 0xe0800000029c181a */
        /*0110*/                ISETP.GE.AND P0, PT, R6, R3, PT;            /* 0xdb681c00019c181e */
        /*0118*/           @!P0 LOP32I.AND R7, R7, 0x4000001f;              /* 0x202000000fa01c1c */
        /*0120*/           @!P0 IMAD R6, R0, c[0x0][0x150], R5;             /* 0x510814002a20001a */
        /*0128*/           @!P0 IADD R6, R6, R7;                            /* 0xe080000003a0181a */
        /*0130*/           @!P0 SHF.L R6, RZ, 0x2, R6;                      /* 0xb7c018000123fc19 */
        /*0138*/                IADD R5, R5, 0x20;                          /* 0xc0800000101c1415 */
                                                                            /* 0x08b8b8b0c8a0b010 */
        /*0148*/           @!P0 BFE R7, R6, 0x11f;                          /* 0xc00800008fa0181d */
        /*0150*/           @!P0 IADD R6.CC, R6, c[0x0][0x168];              /* 0x608400002d20181a */
        /*0158*/           @!P0 IADD.X R7, R7, c[0x0][0x16c];               /* 0x608040002da01c1e */
        /*0160*/           @!P0 ST.E [R6], RZ;                              /* 0xe480000000201bfc */
        /*0168*/                ISETP.LT.AND P0, PT, R5, R3, PT;            /* 0xdb181c00019c141e */
        /*0170*/            @P0 BRA 0xf0;                                   /* 0x12007fffbc00003c */
        /*0178*/                BRA 0x310;                                  /* 0x12000000c81c003c */
                                                                            /* 0x08a0a0a010a01000 */
        /*0188*/                MOV R16, c[0x0][0x140];                     /* 0x64c03c00281c0042 */
        /*0190*/                S2R R10, SR_TID.X;                          /* 0x86400000109c002a */
        /*0198*/                SSY 0x2a0;                                  /* 0x1480000080000000 */
        /*01a0*/                LOP.AND R8, R10, 0x1f;                      /* 0xc20000000f9c2821 */
        /*01a8*/                PSETP.AND.AND P2, PT, PT, PT, PT;           /* 0x84801c07001dc05e */
        /*01b0*/                IADD R12, R8, R5;                           /* 0xe0800000029c2032 */
        /*01b8*/                I2F.F32.S32 R7, R12;                        /* 0xe5c00000061ca81e */
                                                                            /* 0x0880009880108010 */
        /*01c8*/                PSETP.AND.AND P3, PT, P0, PT, PT;           /* 0x84801c07001c007e */
        /*01d0*/                FFMA R11, R7, c[0x0][0x144], R16;           /* 0x4c004000289c1c2e */
        /*01d8*/                PSETP.AND.AND P1, PT, !PT, PT, PT;          /* 0x84801c07001fc03e */
        /*01e0*/                MOV R7, RZ;                                 /* 0xe4c03c007f9c001e */
        /*01e8*/                MOV R8, R6;                                 /* 0xe4c03c00031c0022 */
        /*01f0*/                MOV R9, R11;                                /* 0xe4c03c00059c0026 */
        /*01f8*/                FMUL R14, R9, R9;                           /* 0xe3400000049c243a */
                                                                            /* 0x08b0ac80b0a0a010 */
        /*0208*/                FMUL R15, R8, R8;                           /* 0xe3400000041c203e */
        /*0210*/                PSETP.AND.AND P3, PT, P2, P3, PT;           /* 0x84801c03001c807e */
        /*0218*/                FADD R13, R15, R14;                         /* 0xe2c00000071c3c36 */
        /*0220*/                FSETP.GTU.AND P2, PT, R13, 4, PT;           /* 0xb5e01e04001c345d */
        /*0228*/                PSETP.AND.OR P1, PT, P3, P2, P1;            /* 0x84810402001cc03e */
        /*0230*/                PSETP.AND.AND P2, PT, !PT, PT, PT;          /* 0x84801c07001fc05e */
        /*0238*/                PSETP.XOR.AND P5, PT, P1, P3, PT;           /* 0x84801c03101c40be */
                                                                            /* 0x08ac8010b0a010b0 */
        /*0248*/            @P5 PSETP.AND.AND P2, PT, P3, !P1, PT;          /* 0x84801c090014c05e */
        /*0250*/            @P2 IADD R7, R7, 0x1;                           /* 0xc080000000881c1d */
        /*0258*/            @P5 FADD R13, R9, R9;                           /* 0xe2c0000004942436 */
        /*0260*/                ISETP.LT.AND P3, PT, R7, c[0x0][0x160], PT; /* 0x5b181c002c1c1c7e */
        /*0268*/            @P5 FADD R14, R14, -R15;                        /* 0xe2c100000794383a */
        /*0270*/                PSETP.AND.AND P4, PT, P2, P3, PT;           /* 0x84801c03001c809e */
        /*0278*/            @P5 FFMA R8, R8, R13, R6;                       /* 0xcc00180006942022 */
                                                                            /* 0x08a0a0800000b810 */
        /*0288*/            @P5 FADD R9, R11, R14;                          /* 0xe2c0000007142c26 */
        /*0290*/            @P4 BRA 0x1f8;                                  /* 0x12007fffb010003c */
        /*0298*/                ISETP.GE.AND.S P1, PT, R12, R3, PT;         /* 0xdb681c0001dc303e */
        /*02a0*/            @P1 BRA.U 0x2f0;                                /* 0x120000002404023c */
        /*02a8*/           @!P1 LOP32I.AND R9, R10, 0x4000001f;             /* 0x202000000fa42824 */
        /*02b0*/           @!P1 IMAD R8, R0, c[0x0][0x150], R5;             /* 0x510814002a240022 */
        /*02b8*/           @!P1 IADD R8, R8, R9;                            /* 0xe080000004a42022 */
                                                                            /* 0x08b0a000a0b010a0 */
        /*02c8*/           @!P1 SHF.L R8, RZ, 0x2, R8;                      /* 0xb7c020000127fc21 */
        /*02d0*/           @!P1 BFE R9, R8, 0x11f;                          /* 0xc00800008fa42025 */
        /*02d8*/           @!P1 IADD R8.CC, R8, c[0x0][0x168];              /* 0x608400002d242022 */
        /*02e0*/           @!P1 IADD.X R9, R9, c[0x0][0x16c];               /* 0x608040002da42426 */
        /*02e8*/           @!P1 ST.E [R8], R7;                              /* 0xe48000000024201c */
        /*02f0*/                IADD R5, R5, 0x20;                          /* 0xc0800000101c1415 */
        /*02f8*/                ISETP.LT.AND P1, PT, R5, R3, PT;            /* 0xdb181c00019c143e */
                                                                            /* 0x0800b810b8b000b8 */
        /*0308*/            @P1 BRA 0x190;                                  /* 0x12007fff4004003c */
        /*0310*/                IADD.S R0, R0, 0x1;                         /* 0xc080000000dc0001 */
        /*0318*/                ISETP.NE.AND P0, PT, R0, R4, PT;            /* 0xdb581c00021c001e */
        /*0320*/            @P0 BRA 0xa0;                                   /* 0x12007ffebc00003c */
        /*0328*/                MOV RZ, RZ;                                 /* 0xe4c03c007f9c03fe */
        /*0330*/                EXIT ;                                      /* 0x18000000001c003c */
        /*0338*/                BRA 0x338;                                  /* 0x12007ffffc1c003c */
		....................................


