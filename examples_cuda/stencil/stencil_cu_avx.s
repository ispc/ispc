	.file	"stencil.ispc"
	.text
	.globl	loop_stencil_ispc_tasks
	.align	16, 0x90
	.type	loop_stencil_ispc_tasks,@function
loop_stencil_ispc_tasks:                # @loop_stencil_ispc_tasks
# BB#0:                                 # %allocas
	pushq	%rbp
	movq	%rsp, %rbp
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	andq	$-32, %rsp
	subq	$384, %rsp              # imm = 0x180
	movl	%r9d, 28(%rsp)          # 4-byte Spill
	movl	%r8d, 24(%rsp)          # 4-byte Spill
	movl	%ecx, 20(%rsp)          # 4-byte Spill
	movl	%edx, %ebx
	movl	%esi, 16(%rsp)          # 4-byte Spill
	movl	%edi, %r13d
	movq	$0, 352(%rsp)
	cmpl	%esi, %r13d
	jge	.LBB0_10
# BB#1:                                 # %for_loop.lr.ph
	movl	24(%rbp), %r14d
	movl	16(%rbp), %r15d
	subl	%r15d, %r14d
	leaq	352(%rsp), %rax
	.align	16, 0x90
.LBB0_2:                                # %for_loop
                                        # =>This Inner Loop Header: Depth=1
	movq	%rax, %r12
	movq	%r12, %rdi
	movl	$96, %esi
	movl	$32, %edx
	callq	CUDAAlloc
	testb	$1, %r13b
	jne	.LBB0_4
# BB#3:                                 # %if_then
                                        #   in Loop: Header=BB0_2 Depth=1
	movl	%ebx, 252(%rsp)
	leaq	252(%rsp), %rax
	movq	%rax, 256(%rsp)
	movl	20(%rsp), %eax          # 4-byte Reload
	movl	%eax, 248(%rsp)
	leaq	248(%rsp), %rax
	movq	%rax, 264(%rsp)
	movl	24(%rsp), %eax          # 4-byte Reload
	movl	%eax, 244(%rsp)
	leaq	244(%rsp), %rax
	movq	%rax, 272(%rsp)
	movl	28(%rsp), %eax          # 4-byte Reload
	movl	%eax, 240(%rsp)
	leaq	240(%rsp), %rax
	movq	%rax, 280(%rsp)
	movl	%r15d, 236(%rsp)
	leaq	236(%rsp), %rax
	movq	%rax, 288(%rsp)
	movl	32(%rbp), %eax
	movl	%eax, 232(%rsp)
	leaq	232(%rsp), %rax
	movq	%rax, 296(%rsp)
	movl	40(%rbp), %eax
	movl	%eax, 228(%rsp)
	leaq	228(%rsp), %rax
	movq	%rax, 304(%rsp)
	movl	48(%rbp), %eax
	movl	%eax, 224(%rsp)
	leaq	224(%rsp), %rax
	movq	%rax, 312(%rsp)
	movq	56(%rbp), %rax
	movq	%rax, 216(%rsp)
	leaq	216(%rsp), %rax
	movq	%rax, 320(%rsp)
	movq	64(%rbp), %rax
	movq	%rax, 208(%rsp)
	leaq	208(%rsp), %rax
	movq	%rax, 328(%rsp)
	movq	72(%rbp), %rax
	movq	%rax, 200(%rsp)
	leaq	200(%rsp), %rax
	movq	%rax, 336(%rsp)
	movq	80(%rbp), %rax
	movq	%rax, 192(%rsp)
	leaq	192(%rsp), %rax
	movq	%rax, 344(%rsp)
	movl	$1, 8(%rsp)
	movl	$1, (%rsp)
	movq	%r12, %rdi
	movl	$.L.module_str, %esi
	movl	$.L.ptx_str, %edx
	movl	$.L.func_str, %ecx
	leaq	256(%rsp), %r8
	jmp	.LBB0_5
	.align	16, 0x90
.LBB0_4:                                # %if_else
                                        #   in Loop: Header=BB0_2 Depth=1
	movl	%ebx, 92(%rsp)
	leaq	92(%rsp), %rax
	movq	%rax, 96(%rsp)
	movl	20(%rsp), %eax          # 4-byte Reload
	movl	%eax, 88(%rsp)
	leaq	88(%rsp), %rax
	movq	%rax, 104(%rsp)
	movl	24(%rsp), %eax          # 4-byte Reload
	movl	%eax, 84(%rsp)
	leaq	84(%rsp), %rax
	movq	%rax, 112(%rsp)
	movl	28(%rsp), %eax          # 4-byte Reload
	movl	%eax, 80(%rsp)
	leaq	80(%rsp), %rax
	movq	%rax, 120(%rsp)
	movl	%r15d, 76(%rsp)
	leaq	76(%rsp), %rax
	movq	%rax, 128(%rsp)
	movl	32(%rbp), %eax
	movl	%eax, 72(%rsp)
	leaq	72(%rsp), %rax
	movq	%rax, 136(%rsp)
	movl	40(%rbp), %eax
	movl	%eax, 68(%rsp)
	leaq	68(%rsp), %rax
	movq	%rax, 144(%rsp)
	movl	48(%rbp), %eax
	movl	%eax, 64(%rsp)
	leaq	64(%rsp), %rax
	movq	%rax, 152(%rsp)
	movq	56(%rbp), %rax
	movq	%rax, 56(%rsp)
	leaq	56(%rsp), %rax
	movq	%rax, 160(%rsp)
	movq	64(%rbp), %rax
	movq	%rax, 48(%rsp)
	leaq	48(%rsp), %rax
	movq	%rax, 168(%rsp)
	movq	80(%rbp), %rax
	movq	%rax, 40(%rsp)
	leaq	40(%rsp), %rax
	movq	%rax, 176(%rsp)
	movq	72(%rbp), %rax
	movq	%rax, 32(%rsp)
	leaq	32(%rsp), %rax
	movq	%rax, 184(%rsp)
	movl	$1, 8(%rsp)
	movl	$1, (%rsp)
	movq	%r12, %rdi
	movl	$.L.module_str, %esi
	movl	$.L.ptx_str, %edx
	movl	$.L.func_str1, %ecx
	leaq	96(%rsp), %r8
.LBB0_5:                                # %if_else
                                        #   in Loop: Header=BB0_2 Depth=1
	movl	%r14d, %r9d
	callq	CUDALaunch
	movq	352(%rsp), %rdi
	testq	%rdi, %rdi
	je	.LBB0_7
# BB#6:                                 # %call_sync
                                        #   in Loop: Header=BB0_2 Depth=1
	callq	ISPCSync
	movq	$0, 352(%rsp)
.LBB0_7:                                # %post_sync
                                        #   in Loop: Header=BB0_2 Depth=1
	incl	%r13d
	cmpl	%r13d, 16(%rsp)         # 4-byte Folded Reload
	movq	%r12, %rax
	jne	.LBB0_2
# BB#8:                                 # %for_exit
	movq	352(%rsp), %rdi
	testq	%rdi, %rdi
	je	.LBB0_10
# BB#9:                                 # %call_sync113
	callq	ISPCSync
	movq	$0, 352(%rsp)
.LBB0_10:                               # %post_sync114
	leaq	-40(%rbp), %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	ret
.Ltmp0:
	.size	loop_stencil_ispc_tasks, .Ltmp0-loop_stencil_ispc_tasks

	.type	.L.module_str,@object   # @.module_str
	.section	.rodata,"a",@progbits
.L.module_str:
	.asciz	 "stencil.ispc"
	.size	.L.module_str, 13

	.type	.L.ptx_str,@object      # @.ptx_str
	.align	16
.L.ptx_str:
	.asciz	 "//\n// Generated by LLVM NVPTX Back-End\n//\n\n.version 3.1\n.target sm_35, texmode_independent\n.address_size 64\n\n\t// .globl\tstencil_step_task\n                                        // @stencil_step_task\n.entry stencil_step_task(\n\t.param .u32 stencil_step_task_param_0,\n\t.param .u32 stencil_step_task_param_1,\n\t.param .u32 stencil_step_task_param_2,\n\t.param .u32 stencil_step_task_param_3,\n\t.param .u32 stencil_step_task_param_4,\n\t.param .u32 stencil_step_task_param_5,\n\t.param .u32 stencil_step_task_param_6,\n\t.param .u32 stencil_step_task_param_7,\n\t.param .u64 .ptr .align 8 stencil_step_task_param_8,\n\t.param .u64 .ptr .align 8 stencil_step_task_param_9,\n\t.param .u64 .ptr .align 8 stencil_step_task_param_10,\n\t.param .u64 .ptr .align 8 stencil_step_task_param_11\n)\n{\n\t.reg .pred %p<396>;\n\t.reg .s16 %rc<396>;\n\t.reg .s16 %rs<396>;\n\t.reg .s32 %r<396>;\n\t.reg .s64 %rl<396>;\n\t.reg .f32 %f<396>;\n\t.reg .f64 %fl<396>;\n\n// BB#0:                                // %allocas\n\tmov.u32 \t%r12, %ctaid.x;\n\tld.param.u32 \t%r13, [stencil_step_task_param_4];\n\tadd.s32 \t%r16, %r12, %r13;\n\tadd.s32 \t%r0, %r16, 1;\n\tsetp.ge.s32 \t%p0, %r16, %r0;\n\t@%p0 bra \tBB0_11;\n// BB#1:                                // %for_test28.i.preheader.lr.ph\n\tld.param.u32 \t%r0, [stencil_step_task_param_0];\n\tld.param.u32 \t%r1, [stencil_step_task_param_1];\n\tld.param.u32 \t%r2, [stencil_step_task_param_2];\n\tld.param.u32 \t%r3, [stencil_step_task_param_3];\n\tld.param.u32 \t%r4, [stencil_step_task_param_5];\n\tld.param.u32 \t%r5, [stencil_step_task_param_6];\n\tmul.lo.s32 \t%r5, %r5, %r4;\n\tld.param.u64 \t%rl3, [stencil_step_task_param_8];\n\tld.f64 \t%fl0, [%rl3];\n\tld.f64 \t%fl1, [%rl3+8];\n\tld.param.u64 \t%rl0, [stencil_step_task_param_9];\n\tld.f64 \t%fl2, [%rl3+16];\n\tld.param.u64 \t%rl1, [stencil_step_task_param_10];\n\tld.param.u64 \t%rl2, [stencil_step_task_param_11];\n\tld.f64 \t%fl3, [%rl3+24];\n\tshl.b32 \t%r6, %r4, 1;\n\tmul.lo.s32 \t%r7, %r4, 3;\n\tmul.lo.s32 \t%r8, %r4, -3;\n\tshl.b32 \t%r9, %r5, 1;\n\tmul.lo.s32 \t%r10, %r5, 3;\n\tmul.lo.s32 \t%r11, %r5, -3;\n\tadd.s32 \t%r12, %r12, %r13;\n\tneg.s32 \t%r13, %r9;\n\tneg.s32 \t%r14, %r6;\n\tmov.u32 \t%r32, WARP_SZ;\nBB0_2:                                  // %for_test28.i.preheader\n                                        // =>This Loop Header: Depth=1\n                                        //     Child Loop BB0_9 Depth 2\n                                        //       Child Loop BB0_5 Depth 3\n\tmov.u32 \t%r15, %r16;\n\tsetp.ge.s32 \t%p0, %r2, %r3;\n\t@%p0 bra \tBB0_10;\n// BB#3:                                // %for_test35.i.preheader.lr.ph\n                                        //   in Loop: Header=BB0_2 Depth=1\n\tsetp.lt.s32 \t%p0, %r0, %r1;\n\t@%p0 bra \tBB0_4;\n\tbra.uni \tBB0_10;\nBB0_4:                                  //   in Loop: Header=BB0_2 Depth=1\n\tmul.lo.s32 \t%r16, %r15, %r5;\n\tmov.u32 \t%r17, %r2;\nBB0_9:                                  // %for_loop37.i.lr.ph.us\n                                        //   Parent Loop BB0_2 Depth=1\n                                        // =>  This Loop Header: Depth=2\n                                        //       Child Loop BB0_5 Depth 3\n\tmad.lo.s32 \t%r18, %r17, %r4, %r16;\n\tadd.s32 \t%r19, %r18, %r4;\n\tadd.s32 \t%r20, %r18, %r6;\n\tsub.s32 \t%r21, %r18, %r4;\n\tadd.s32 \t%r22, %r18, %r7;\n\tadd.s32 \t%r23, %r18, %r14;\n\tadd.s32 \t%r24, %r18, %r5;\n\tadd.s32 \t%r25, %r18, %r8;\n\tadd.s32 \t%r26, %r18, %r9;\n\tsub.s32 \t%r27, %r18, %r5;\n\tadd.s32 \t%r28, %r18, %r10;\n\tadd.s32 \t%r29, %r18, %r13;\n\tadd.s32 \t%r30, %r18, %r11;\n\tmov.u32 \t%r31, %r0;\nBB0_5:                                  // %for_loop37.i.us\n                                        //   Parent Loop BB0_2 Depth=1\n                                        //     Parent Loop BB0_9 Depth=2\n                                        // =>    This Inner Loop Header: Depth=3\n\tmov.u32 \t%r33, %tid.x;\n\tadd.s32 \t%r34, %r32, -1;\n\tand.b32  \t%r33, %r34, %r33;\n\tadd.s32 \t%r33, %r33, %r31;\n\tsetp.ge.s32 \t%p0, %r33, %r1;\n\t@%p0 bra \tBB0_7;\n// BB#6:                                // %pl_dolane.i.us\n                                        //   in Loop: Header=BB0_5 Depth=3\n\tadd.s32 \t%r34, %r18, %r33;\n\tshl.b32 \t%r34, %r34, 3;\n\tadd.s32 \t%r35, %r34, -8;\n\tcvt.s64.s32 \t%rl3, %r35;\n\tadd.s64 \t%rl3, %rl3, %rl1;\n\tld.f64 \t%fl4, [%rl3];\n\tadd.s32 \t%r35, %r34, 8;\n\tcvt.s64.s32 \t%rl3, %r35;\n\tadd.s64 \t%rl3, %rl3, %rl1;\n\tld.f64 \t%fl5, [%rl3];\n\tadd.s32 \t%r35, %r34, -16;\n\tcvt.s64.s32 \t%rl3, %r35;\n\tadd.s64 \t%rl3, %rl3, %rl1;\n\tld.f64 \t%fl6, [%rl3];\n\tadd.s32 \t%r35, %r34, 16;\n\tcvt.s64.s32 \t%rl3, %r35;\n\tadd.s64 \t%rl3, %rl3, %rl1;\n\tld.f64 \t%fl9, [%rl3];\n\tadd.s32 \t%r35, %r19, %r33;\n\tshl.b32 \t%r35, %r35, 3;\n\tcvt.s64.s32 \t%rl3, %r35;\n\tadd.s64 \t%rl3, %rl3, %rl1;\n\tld.f64 \t%fl8, [%rl3];\n\tadd.s32 \t%r35, %r34, -24;\n\tcvt.s64.s32 \t%rl3, %r35;\n\tadd.s64 \t%rl3, %rl3, %rl1;\n\tld.f64 \t%fl7, [%rl3];\n\tadd.s32 \t%r35, %r34, 24;\n\tcvt.s64.s32 \t%rl3, %r35;\n\tadd.s64 \t%rl3, %rl3, %rl1;\n\tld.f64 \t%fl10, [%rl3];\n\tadd.s32 \t%r35, %r20, %r33;\n\tshl.b32 \t%r35, %r35, 3;\n\tcvt.s64.s32 \t%rl3, %r35;\n\tadd.s64 \t%rl3, %rl3, %rl1;\n\tld.f64 \t%fl13, [%rl3];\n\tadd.s32 \t%r35, %r21, %r33;\n\tshl.b32 \t%r35, %r35, 3;\n\tcvt.s64.s32 \t%rl3, %r35;\n\tadd.s64 \t%rl3, %rl3, %rl1;\n\tld.f64 \t%fl12, [%rl3];\n\tadd.s32 \t%r35, %r22, %r33;\n\tshl.b32 \t%r35, %r35, 3;\n\tcvt.s64.s32 \t%rl3, %r35;\n\tadd.s64 \t%rl3, %rl3, %rl1;\n\tld.f64 \t%fl11, [%rl3];\n\tadd.s32 \t%r35, %r23, %r33;\n\tshl.b32 \t%r35, %r35, 3;\n\tcvt.s64.s32 \t%rl3, %r35;\n\tadd.s64 \t%rl3, %rl3, %rl1;\n\tld.f64 \t%fl16, [%rl3];\n\tadd.s32 \t%r35, %r24, %r33;\n\tshl.b32 \t%r35, %r35, 3;\n\tcvt.s64.s32 \t%rl3, %r35;\n\tadd.s64 \t%rl3, %rl3, %rl1;\n\tld.f64 \t%fl15, [%rl3];\n\tadd.s32 \t%r35, %r25, %r33;\n\tshl.b32 \t%r35, %r35, 3;\n\tcvt.s64.s32 \t%rl3, %r35;\n\tadd.s64 \t%rl3, %rl3, %rl1;\n\tld.f64 \t%fl14, [%rl3];\n\tadd.s32 \t%r35, %r26, %r33;\n\tshl.b32 \t%r35, %r35, 3;\n\tcvt.s64.s32 \t%rl3, %r35;\n\tadd.s64 \t%rl3, %rl3, %rl1;\n\tld.f64 \t%fl19, [%rl3];\n\tadd.s32 \t%r35, %r27, %r33;\n\tshl.b32 \t%r35, %r35, 3;\n\tcvt.s64.s32 \t%rl3, %r35;\n\tadd.s64 \t%rl3, %rl3, %rl1;\n\tld.f64 \t%fl18, [%rl3];\n\tadd.s32 \t%r35, %r28, %r33;\n\tshl.b32 \t%r35, %r35, 3;\n\tcvt.s64.s32 \t%rl3, %r35;\n\tadd.s64 \t%rl3, %rl3, %rl1;\n\tld.f64 \t%fl17, [%rl3];\n\tadd.s32 \t%r35, %r29, %r33;\n\tshl.b32 \t%r35, %r35, 3;\n\tcvt.s64.s32 \t%rl3, %r35;\n\tadd.s64 \t%rl3, %rl3, %rl1;\n\tld.f64 \t%fl24, [%rl3];\n\tcvt.s64.s32 \t%rl4, %r34;\n\tadd.s64 \t%rl3, %rl4, %rl1;\n\tld.f64 \t%fl21, [%rl3];\n\tadd.s32 \t%r33, %r30, %r33;\n\tshl.b32 \t%r33, %r33, 3;\n\tcvt.s64.s32 \t%rl3, %r33;\n\tadd.s64 \t%rl3, %rl3, %rl1;\n\tld.f64 \t%fl20, [%rl3];\n\tadd.s64 \t%rl3, %rl4, %rl2;\n\tld.f64 \t%fl23, [%rl3];\n\tadd.s64 \t%rl4, %rl4, %rl0;\n\tld.f64 \t%fl22, [%rl4];\n\tadd.f64 \t%fl25, %fl21, %fl21;\n\tsub.f64 \t%fl23, %fl25, %fl23;\n\tadd.f64 \t%fl6, %fl6, %fl9;\n\tadd.f64 \t%fl6, %fl6, %fl13;\n\tadd.f64 \t%fl6, %fl6, %fl16;\n\tadd.f64 \t%fl6, %fl6, %fl19;\n\tadd.f64 \t%fl6, %fl6, %fl24;\n\tadd.f64 \t%fl4, %fl4, %fl5;\n\tadd.f64 \t%fl4, %fl4, %fl8;\n\tadd.f64 \t%fl4, %fl4, %fl12;\n\tadd.f64 \t%fl4, %fl4, %fl15;\n\tadd.f64 \t%fl4, %fl4, %fl18;\n\tmul.f64 \t%fl5, %fl0, %fl21;\n\tfma.rn.f64 \t%fl4, %fl1, %fl4, %fl5;\n\tfma.rn.f64 \t%fl4, %fl2, %fl6, %fl4;\n\tadd.f64 \t%fl5, %fl7, %fl10;\n\tadd.f64 \t%fl5, %fl5, %fl11;\n\tadd.f64 \t%fl5, %fl5, %fl14;\n\tadd.f64 \t%fl5, %fl5, %fl17;\n\tadd.f64 \t%fl5, %fl5, %fl20;\n\tfma.rn.f64 \t%fl4, %fl3, %fl5, %fl4;\n\tfma.rn.f64 \t%fl4, %fl4, %fl22, %fl23;\n\tst.f64 \t[%rl3], %fl4;\nBB0_7:                                  // %safe_if_after_true.i.us\n                                        //   in Loop: Header=BB0_5 Depth=3\n\tadd.s32 \t%r31, %r32, %r31;\n\tsetp.lt.s32 \t%p0, %r31, %r1;\n\t@%p0 bra \tBB0_5;\n// BB#8:                                // %for_exit38.i.us\n                                        //   in Loop: Header=BB0_9 Depth=2\n\tadd.s32 \t%r17, %r17, 1;\n\tsetp.eq.s32 \t%p0, %r17, %r3;\n\t@%p0 bra \tBB0_10;\n\tbra.uni \tBB0_9;\nBB0_10:                                 // %for_exit31.i\n                                        //   in Loop: Header=BB0_2 Depth=1\n\tadd.s32 \t%r16, %r15, 1;\n\tsetp.ne.s32 \t%p0, %r15, %r12;\n\t@%p0 bra \tBB0_2;\nBB0_11:                                 // %stencil_step___uniuniuniuniuniuniuniuniuniun_3C_Cund_3E_un_3C_Cund_3E_un_3C_Cund_3E_un_3C_und_3E_.exit\n\tret;\n}\n\n"
	.size	.L.ptx_str, 7954

	.type	.L.func_str,@object     # @.func_str
	.align	16
.L.func_str:
	.asciz	 "stencil_step_task"
	.size	.L.func_str, 18

	.type	.L.func_str1,@object    # @.func_str1
	.align	16
.L.func_str1:
	.asciz	 "stencil_step_task"
	.size	.L.func_str1, 18


	.section	".note.GNU-stack","",@progbits
