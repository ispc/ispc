	.file	"stencil_cu.ll"
	.section	.rodata.cst16,"aM",@progbits,16
	.align	16
.LCPI0_0:
	.long	4                       # 0x4
	.long	5                       # 0x5
	.long	6                       # 0x6
	.long	7                       # 0x7
.LCPI0_1:
	.long	0                       # 0x0
	.long	1                       # 0x1
	.long	2                       # 0x2
	.long	3                       # 0x3
	.section	.rodata,"a",@progbits
	.align	32
.LCPI0_2:
	.quad	4611686018427387904     # double 2.000000e+00
	.quad	4611686018427387904     # double 2.000000e+00
	.quad	4611686018427387904     # double 2.000000e+00
	.quad	4611686018427387904     # double 2.000000e+00
	.text
	.align	16, 0x90
	.type	stencil_step___uniuniuniuniuniuniuniuniuniun_3C_Cund_3E_un_3C_Cund_3E_un_3C_Cund_3E_un_3C_und_3E_,@function
stencil_step___uniuniuniuniuniuniuniuniuniun_3C_Cund_3E_un_3C_Cund_3E_un_3C_Cund_3E_un_3C_und_3E_: # @stencil_step___uniuniuniuniuniuniuniuniuniun_3C_Cund_3E_un_3C_Cund_3E_un_3C_Cund_3E_un_3C_und_3E_
# BB#0:                                 # %allocas
	pushq	%rbp
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	subq	$1384, %rsp             # imm = 0x568
	movl	%ecx, -72(%rsp)         # 4-byte Spill
	movl	%esi, 1308(%rsp)        # 4-byte Spill
	movl	%edi, -68(%rsp)         # 4-byte Spill
	movq	1456(%rsp), %rcx
	vmovsd	24(%rcx), %xmm1
	vmovsd	16(%rcx), %xmm3
	movq	1472(%rsp), %rax
	vmovsd	(%rcx), %xmm2
	vmovsd	8(%rcx), %xmm4
	movl	1448(%rsp), %esi
	vmovmskps	%ymm0, %ecx
	cmpl	$255, %ecx
	jne	.LBB0_1
# BB#7:                                 # %for_test.preheader
	cmpl	%r9d, %r8d
	jge	.LBB0_6
# BB#8:                                 # %for_test30.preheader.lr.ph
	leal	-3(%r8), %ecx
	leal	2(%r8), %r13d
	leal	-1(%r8), %edi
	leal	3(%r8), %ebp
	movl	%esi, %r11d
	imull	%r11d, %ebp
	movl	%ebp, %ebx
	imull	%r11d, %edi
	movl	%edi, %ebp
	imull	%r11d, %r13d
	imull	%r8d, %esi
	imull	%r11d, %ecx
	leal	-2(%r8), %r10d
	imull	%r11d, %r10d
	leal	1(%r8), %r14d
	imull	%r11d, %r14d
	movl	%edx, -96(%rsp)         # 4-byte Spill
	addl	%edx, %r14d
	addl	%edx, %r10d
	addl	%edx, %ecx
	movl	%ecx, 1344(%rsp)        # 4-byte Spill
	movl	%r9d, -92(%rsp)         # 4-byte Spill
	leal	1(%rdx,%rsi), %r15d
	leal	2(%rdx,%rsi), %edi
	addl	%edx, %r13d
	addl	%edx, %ebp
	movl	%ebp, 1216(%rsp)        # 4-byte Spill
	addl	%edx, %ebx
	movl	%ebx, 1152(%rsp)        # 4-byte Spill
	leal	-1(%rdx,%rsi), %ebp
	leal	3(%rdx,%rsi), %ecx
	leal	(%rdx,%rsi), %r12d
	leal	-3(%rdx,%rsi), %ebx
	movl	%ebx, 1184(%rsp)        # 4-byte Spill
	movl	%r8d, -88(%rsp)         # 4-byte Spill
	leal	-2(%rdx,%rsi), %edx
	vmovd	1308(%rsp), %xmm0       # 4-byte Folded Reload
	movl	1440(%rsp), %r9d
	imull	%r9d, %r13d
	imull	%r9d, %ecx
	movl	%ecx, 1312(%rsp)        # 4-byte Spill
	imull	%r9d, %ebp
	movl	%ebp, 1248(%rsp)        # 4-byte Spill
	imull	%r9d, %edi
	imull	%r9d, %r15d
	movl	1344(%rsp), %ecx        # 4-byte Reload
	imull	%r9d, %ecx
	movl	%ecx, 1344(%rsp)        # 4-byte Spill
	imull	%r9d, %r10d
	movl	1152(%rsp), %ebx        # 4-byte Reload
	imull	%r9d, %ebx
	movl	1216(%rsp), %ebp        # 4-byte Reload
	imull	%r9d, %ebp
	imull	%r9d, %r14d
	movl	1184(%rsp), %r8d        # 4-byte Reload
	imull	%r9d, %r8d
	imull	%r9d, %edx
	movl	%edx, 1216(%rsp)        # 4-byte Spill
	imull	%r9d, %r12d
	movl	-68(%rsp), %edx         # 4-byte Reload
	leal	(,%rdx,8), %edx
	leal	-16(%rdx,%r12,8), %esi
	movl	%esi, 76(%rsp)          # 4-byte Spill
	leal	(%rdx,%r12,8), %ecx
	movl	%ecx, 72(%rsp)          # 4-byte Spill
	leal	(%rdx,%r15,8), %ecx
	movl	%ecx, 68(%rsp)          # 4-byte Spill
	movl	-92(%rsp), %ecx         # 4-byte Reload
	leal	(%rdx,%rdi,8), %esi
	movl	%esi, 64(%rsp)          # 4-byte Spill
	movl	1248(%rsp), %esi        # 4-byte Reload
	leal	(%rdx,%rsi,8), %esi
	movl	%esi, 60(%rsp)          # 4-byte Spill
	movl	1312(%rsp), %esi        # 4-byte Reload
	leal	(%rdx,%rsi,8), %esi
	movl	%esi, 56(%rsp)          # 4-byte Spill
	movl	1216(%rsp), %esi        # 4-byte Reload
	leal	(%rdx,%rsi,8), %esi
	movl	%esi, 52(%rsp)          # 4-byte Spill
	movl	-88(%rsp), %esi         # 4-byte Reload
	leal	(%rdx,%r8,8), %edi
	movl	%edi, 48(%rsp)          # 4-byte Spill
	leal	(%rdx,%r14,8), %edi
	movl	%edi, 44(%rsp)          # 4-byte Spill
	leal	(%rdx,%r13,8), %edi
	movl	%edi, 40(%rsp)          # 4-byte Spill
	leal	(%rdx,%rbp,8), %edi
	movl	%edi, 36(%rsp)          # 4-byte Spill
	leal	(%rdx,%rbx,8), %edi
	movl	%edi, 32(%rsp)          # 4-byte Spill
	leal	(%rdx,%r10,8), %edi
	movl	%edi, 28(%rsp)          # 4-byte Spill
	movl	1344(%rsp), %edi        # 4-byte Reload
	leal	(%rdx,%rdi,8), %edx
	movl	%edx, 24(%rsp)          # 4-byte Spill
	movl	$0, -100(%rsp)          # 4-byte Folded Spill
	imull	%r9d, %r11d
	shll	$3, %r9d
	movl	%r9d, -76(%rsp)         # 4-byte Spill
	shll	$3, %r11d
	movl	%r11d, -104(%rsp)       # 4-byte Spill
	vpermilpd	$0, %xmm3, %xmm3 # xmm3 = xmm3[0,0]
	vpermilpd	$0, %xmm2, %xmm2 # xmm2 = xmm2[0,0]
	vpermilpd	$0, %xmm1, %xmm1 # xmm1 = xmm1[0,0]
	vpshufd	$0, %xmm0, %xmm0        # xmm0 = xmm0[0,0,0,0]
	vinsertf128	$1, %xmm1, %ymm1, %ymm1
	vmovupd	%ymm1, 1312(%rsp)       # 32-byte Folded Spill
	vinsertf128	$1, %xmm3, %ymm3, %ymm1
	vmovupd	%ymm1, 1344(%rsp)       # 32-byte Folded Spill
	vinsertf128	$1, %xmm2, %ymm2, %ymm15
	vmovupd	%ymm15, -32(%rsp)       # 32-byte Folded Spill
	vpermilpd	$0, %xmm4, %xmm1 # xmm1 = xmm4[0,0]
	vinsertf128	$1, %xmm1, %ymm1, %ymm14
	vmovupd	%ymm14, -64(%rsp)       # 32-byte Folded Spill
	vinsertf128	$1, %xmm0, %ymm0, %ymm0
	vmovups	%ymm0, 1248(%rsp)       # 32-byte Folded Spill
	vmovapd	.LCPI0_2(%rip), %ymm13
	.align	16, 0x90
.LBB0_9:                                # %for_test30.preheader
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_16 Depth 2
                                        #       Child Loop BB0_12 Depth 3
	movl	%esi, -88(%rsp)         # 4-byte Spill
	movl	-96(%rsp), %edx         # 4-byte Reload
	cmpl	-72(%rsp), %edx         # 4-byte Folded Reload
	jge	.LBB0_11
# BB#10:                                # %for_test37.preheader.lr.ph
                                        #   in Loop: Header=BB0_9 Depth=1
	movl	-68(%rsp), %edx         # 4-byte Reload
	cmpl	1308(%rsp), %edx        # 4-byte Folded Reload
	movl	-100(%rsp), %edx        # 4-byte Reload
	movl	-96(%rsp), %edi         # 4-byte Reload
	jge	.LBB0_11
	.align	16, 0x90
.LBB0_16:                               # %for_loop39.lr.ph.us
                                        #   Parent Loop BB0_9 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_12 Depth 3
	movl	%edi, -84(%rsp)         # 4-byte Spill
	movl	%edx, -80(%rsp)         # 4-byte Spill
	movl	%edx, %r13d
	movl	-68(%rsp), %ecx         # 4-byte Reload
	.align	16, 0x90
.LBB0_12:                               # %for_loop39.us
                                        #   Parent Loop BB0_9 Depth=1
                                        #     Parent Loop BB0_16 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	movl	%ecx, 1216(%rsp)        # 4-byte Spill
	vmovups	1248(%rsp), %ymm3       # 32-byte Folded Reload
	vmovups	%ymm3, 1248(%rsp)       # 32-byte Folded Spill
	vextractf128	$1, %ymm3, %xmm0
	vmovd	%ecx, %xmm1
	vpshufd	$0, %xmm1, %xmm1        # xmm1 = xmm1[0,0,0,0]
	vpaddd	.LCPI0_0(%rip), %xmm1, %xmm2
	vpcmpgtd	%xmm2, %xmm0, %xmm0
	vpaddd	.LCPI0_1(%rip), %xmm1, %xmm1
	vpcmpgtd	%xmm1, %xmm3, %xmm1
	vinsertf128	$1, %xmm0, %ymm1, %ymm8
	vmovmskps	%ymm8, %ecx
	testl	%ecx, %ecx
	je	.LBB0_14
# BB#13:                                # %safe_if_run_true.us
                                        #   in Loop: Header=BB0_12 Depth=3
	movl	76(%rsp), %esi          # 4-byte Reload
	leal	8(%rsi,%r13), %edx
	movl	68(%rsp), %ecx          # 4-byte Reload
	leal	(%rcx,%r13), %ecx
	movl	72(%rsp), %r12d         # 4-byte Reload
	leal	24(%r12,%r13), %r14d
	leal	-8(%rsi,%r13), %r8d
	movl	52(%rsp), %edi          # 4-byte Reload
	leal	(%rdi,%r13), %edi
	leal	8(%r12,%r13), %ebp
	leal	(%rsi,%r13), %esi
	leal	16(%r12,%r13), %r11d
	movl	64(%rsp), %ebx          # 4-byte Reload
	leal	(%rbx,%r13), %r9d
	movl	44(%rsp), %ebx          # 4-byte Reload
	leal	(%rbx,%r13), %r15d
	movl	60(%rsp), %ebx          # 4-byte Reload
	leal	(%rbx,%r13), %r10d
	movl	40(%rsp), %ebx          # 4-byte Reload
	leal	(%rbx,%r13), %ebx
	movl	%ebx, 832(%rsp)         # 4-byte Spill
	movl	56(%rsp), %ebx          # 4-byte Reload
	leal	(%rbx,%r13), %ebx
	movl	%ebx, 800(%rsp)         # 4-byte Spill
	movl	36(%rsp), %ebx          # 4-byte Reload
	leal	(%rbx,%r13), %ebx
	movl	%ebx, 768(%rsp)         # 4-byte Spill
	movl	28(%rsp), %ebx          # 4-byte Reload
	leal	(%rbx,%r13), %ebx
	movl	%ebx, 736(%rsp)         # 4-byte Spill
	leal	(%r12,%r13), %ebx
	movl	%ebx, 960(%rsp)         # 4-byte Spill
	movl	48(%rsp), %ebx          # 4-byte Reload
	leal	(%rbx,%r13), %ebx
	movl	%ebx, 896(%rsp)         # 4-byte Spill
	movl	32(%rsp), %ebx          # 4-byte Reload
	leal	(%rbx,%r13), %r12d
	movl	24(%rsp), %ebx          # 4-byte Reload
	leal	(%rbx,%r13), %ebx
	movl	%ebx, 992(%rsp)         # 4-byte Spill
	movslq	%edx, %rdx
	movq	%rdx, 1184(%rsp)        # 8-byte Spill
	movslq	%ecx, %rbx
	movq	%rbx, 1056(%rsp)        # 8-byte Spill
	movslq	%esi, %rcx
	movq	%rcx, 1120(%rsp)        # 8-byte Spill
	vmovupd	(%rax,%rbx), %xmm0
	movq	%rbx, %rsi
	vmovupd	16(%rax,%rdx), %xmm2
	vmovupd	(%rax,%rdx), %xmm3
	movslq	%ebp, %rdx
	movq	%rdx, 1152(%rsp)        # 8-byte Spill
	vmovupd	16(%rax,%rdx), %xmm1
	vmovupd	(%rax,%rdx), %xmm4
	vinsertf128	$1, %xmm1, %ymm4, %ymm1
	vinsertf128	$1, %xmm2, %ymm3, %ymm2
	movslq	%edi, %rdx
	movq	%rdx, 928(%rsp)         # 8-byte Spill
	movslq	%r8d, %rbx
	movslq	%r14d, %r14
	vmovupd	16(%rax,%rsi), %xmm3
	vmovupd	16(%rax,%rcx), %xmm4
	vmovupd	(%rax,%rcx), %xmm5
	movslq	%r11d, %rcx
	movq	%rcx, 1088(%rsp)        # 8-byte Spill
	vmovupd	16(%rax,%rcx), %xmm6
	vmovupd	(%rax,%rcx), %xmm7
	vinsertf128	$1, %xmm6, %ymm7, %ymm6
	vinsertf128	$1, %xmm4, %ymm5, %ymm7
	vaddpd	%ymm1, %ymm2, %ymm1
	vinsertf128	$1, %xmm3, %ymm0, %ymm3
	movslq	%r10d, %rsi
	movq	%rsi, 864(%rsp)         # 8-byte Spill
	vmovupd	(%rax,%r14), %xmm5
	vmovupd	(%rax,%rbx), %xmm4
	vmovupd	(%rax,%rdx), %xmm2
	movslq	%r15d, %rbp
	movslq	%r9d, %rcx
	movq	%rcx, 1048(%rsp)        # 8-byte Spill
	vmovupd	16(%rax,%rcx), %xmm0
	vmovupd	(%rax,%rcx), %xmm9
	vaddpd	%ymm6, %ymm7, %ymm7
	vinsertf128	$1, %xmm0, %ymm9, %ymm9
	vmovupd	(%rax,%rbp), %xmm12
	vmovupd	(%rax,%rsi), %xmm6
	vmovupd	16(%rax,%rdx), %xmm0
	vaddpd	%ymm3, %ymm1, %ymm3
	vinsertf128	$1, 16(%rax,%rsi), %ymm6, %ymm6
	vinsertf128	$1, 16(%rax,%r14), %ymm5, %ymm5
	vinsertf128	$1, 16(%rax,%rbx), %ymm4, %ymm4
	vaddpd	%ymm9, %ymm7, %ymm1
	vinsertf128	$1, %xmm0, %ymm2, %ymm2
	movslq	736(%rsp), %r8          # 4-byte Folded Reload
	movslq	768(%rsp), %rdx         # 4-byte Folded Reload
	movslq	800(%rsp), %rdi         # 4-byte Folded Reload
	vmovupd	(%rax,%rdi), %xmm10
	movslq	832(%rsp), %r15         # 4-byte Folded Reload
	vmovupd	(%rax,%r15), %xmm9
	vmovupd	(%rax,%rdx), %xmm7
	vaddpd	%ymm5, %ymm4, %ymm4
	vmovupd	(%rax,%r8), %xmm11
	vaddpd	%ymm6, %ymm3, %ymm5
	vinsertf128	$1, 16(%rax,%rdi), %ymm10, %ymm3
	vinsertf128	$1, 16(%rax,%rbp), %ymm12, %ymm10
	vinsertf128	$1, 16(%rax,%r15), %ymm9, %ymm0
	movslq	896(%rsp), %r11         # 4-byte Folded Reload
	vaddpd	%ymm2, %ymm1, %ymm1
	movslq	960(%rsp), %rcx         # 4-byte Folded Reload
	vmovupd	(%rax,%rcx), %xmm6
	vaddpd	%ymm0, %ymm1, %ymm1
	vinsertf128	$1, 16(%rax,%r8), %ymm11, %ymm2
	vinsertf128	$1, 16(%rax,%rdx), %ymm7, %ymm0
	movslq	%r12d, %r12
	vaddpd	%ymm10, %ymm5, %ymm7
	vmovupd	(%rax,%r11), %xmm5
	vaddpd	%ymm3, %ymm4, %ymm3
	vinsertf128	$1, 16(%rax,%r11), %ymm5, %ymm4
	vinsertf128	$1, 16(%rax,%rcx), %ymm6, %ymm9
	vmovupd	(%rax,%r12), %xmm5
	movslq	992(%rsp), %rsi         # 4-byte Folded Reload
	vaddpd	%ymm0, %ymm7, %ymm10
	vextractf128	$1, %ymm8, %xmm6
	vaddpd	%ymm2, %ymm1, %ymm2
	vpshufd	$80, %xmm6, %xmm7       # xmm7 = xmm6[0,0,1,1]
	vmulpd	%ymm9, %ymm15, %ymm1
	vmovupd	(%rax,%rsi), %xmm9
	vaddpd	%ymm4, %ymm3, %ymm3
	vinsertf128	$1, 16(%rax,%r12), %ymm5, %ymm4
	vpshufd	$80, %xmm8, %xmm5       # xmm5 = xmm8[0,0,1,1]
	vpshufd	$-6, %xmm6, %xmm0       # xmm0 = xmm6[2,2,3,3]
	vpshufd	$-6, %xmm8, %xmm6       # xmm6 = xmm8[2,2,3,3]
	vinsertf128	$1, %xmm6, %ymm5, %ymm6
	vinsertf128	$1, 16(%rax,%rsi), %ymm9, %ymm5
	vinsertf128	$1, %xmm0, %ymm7, %ymm8
	vmovupd	%ymm8, 96(%rsp)         # 32-byte Folded Spill
	vmovupd	1344(%rsp), %ymm0       # 32-byte Folded Reload
	vmovupd	%ymm0, 1344(%rsp)       # 32-byte Folded Spill
	vmovupd	%ymm0, 1344(%rsp)       # 32-byte Folded Spill
	vmulpd	%ymm2, %ymm0, %ymm0
	vmulpd	%ymm10, %ymm14, %ymm2
	movq	1480(%rsp), %r9
	vmaskmovpd	(%r9,%rcx), %ymm6, %ymm7
	vaddpd	%ymm1, %ymm2, %ymm1
	vaddpd	%ymm1, %ymm0, %ymm0
	vaddpd	%ymm4, %ymm3, %ymm3
	vmaskmovpd	(%rax,%rcx), %ymm6, %ymm1
	vmulpd	%ymm13, %ymm1, %ymm1
	movq	1464(%rsp), %r10
	vmaskmovpd	(%r10,%rcx), %ymm6, %ymm2
	vsubpd	%ymm7, %ymm1, %ymm1
	vmaskmovpd	32(%r10,%rcx), %ymm8, %ymm4
	vmovupd	%ymm4, 992(%rsp)        # 32-byte Folded Spill
	vaddpd	%ymm5, %ymm3, %ymm3
	vmovups	48(%rax,%rsi), %xmm4
	vmovaps	%xmm4, 960(%rsp)        # 16-byte Spill
	vmovupd	1312(%rsp), %ymm4       # 32-byte Folded Reload
	vmovupd	%ymm4, 1312(%rsp)       # 32-byte Folded Spill
	vmovupd	%ymm4, 1312(%rsp)       # 32-byte Folded Spill
	vmulpd	%ymm3, %ymm4, %ymm3
	vmovups	32(%rax,%rsi), %xmm4
	vmovups	%ymm4, 896(%rsp)        # 32-byte Folded Spill
	vaddpd	%ymm3, %ymm0, %ymm0
	vmovups	48(%rax,%r12), %xmm3
	vmovaps	%xmm3, 832(%rsp)        # 16-byte Spill
	vmulpd	%ymm2, %ymm0, %ymm0
	vmovups	32(%rax,%r12), %xmm2
	vmovups	%ymm2, 800(%rsp)        # 32-byte Folded Spill
	vaddpd	%ymm0, %ymm1, %ymm0
	vmovupd	%ymm0, 128(%rsp)        # 32-byte Folded Spill
	vmovups	48(%rax,%r11), %xmm0
	vmovaps	%xmm0, 768(%rsp)        # 16-byte Spill
	vmovups	32(%rax,%r11), %xmm0
	vmovups	%ymm0, 736(%rsp)        # 32-byte Folded Spill
	vmovups	48(%rax,%rdi), %xmm0
	vmovaps	%xmm0, 704(%rsp)        # 16-byte Spill
	vmovups	32(%rax,%rdi), %xmm0
	vmovups	%ymm0, 640(%rsp)        # 32-byte Folded Spill
	vmovups	48(%rax,%rbx), %xmm0
	vmovaps	%xmm0, 592(%rsp)        # 16-byte Spill
	vmovups	32(%rax,%rbx), %xmm0
	vmovups	%ymm0, 544(%rsp)        # 32-byte Folded Spill
	vmovups	48(%rax,%r14), %xmm0
	vmovaps	%xmm0, 464(%rsp)        # 16-byte Spill
	vmovups	32(%rax,%r14), %xmm0
	vmovups	%ymm0, 416(%rsp)        # 32-byte Folded Spill
	vmovups	48(%rax,%rdx), %xmm0
	vmovaps	%xmm0, 400(%rsp)        # 16-byte Spill
	vmovups	32(%rax,%rdx), %xmm0
	vmovups	%ymm0, 352(%rsp)        # 32-byte Folded Spill
	vmovups	48(%rax,%r8), %xmm0
	vmovaps	%xmm0, 336(%rsp)        # 16-byte Spill
	vmovups	32(%rax,%r8), %xmm0
	vmovups	%ymm0, 288(%rsp)        # 32-byte Folded Spill
	vmovups	48(%rax,%rbp), %xmm0
	vmovaps	%xmm0, 272(%rsp)        # 16-byte Spill
	vmovups	32(%rax,%rbp), %xmm0
	vmovups	%ymm0, 224(%rsp)        # 32-byte Folded Spill
	vmaskmovpd	32(%r9,%rcx), %ymm8, %ymm0
	vmovupd	%ymm0, 672(%rsp)        # 32-byte Folded Spill
	vmaskmovpd	32(%rax,%rcx), %ymm8, %ymm0
	vmovupd	%ymm0, 608(%rsp)        # 32-byte Folded Spill
	vmovups	48(%rax,%rcx), %xmm0
	vmovaps	%xmm0, 528(%rsp)        # 16-byte Spill
	vmovups	32(%rax,%rcx), %xmm0
	vmovups	%ymm0, 480(%rsp)        # 32-byte Folded Spill
	vmovups	48(%rax,%r15), %xmm0
	vmovaps	%xmm0, 208(%rsp)        # 16-byte Spill
	vmovups	32(%rax,%r15), %xmm0
	vmovups	%ymm0, 160(%rsp)        # 32-byte Folded Spill
	movq	864(%rsp), %rdx         # 8-byte Reload
	vmovups	48(%rax,%rdx), %xmm0
	vmovaps	%xmm0, 80(%rsp)         # 16-byte Spill
	vmovups	32(%rax,%rdx), %xmm0
	vmovups	%ymm0, 864(%rsp)        # 32-byte Folded Spill
	movq	928(%rsp), %rdx         # 8-byte Reload
	vmovupd	48(%rax,%rdx), %xmm4
	vmovupd	32(%rax,%rdx), %xmm9
	movq	1056(%rsp), %rdx        # 8-byte Reload
	vmovupd	48(%rax,%rdx), %xmm5
	vmovupd	32(%rax,%rdx), %xmm11
	movq	1048(%rsp), %rdx        # 8-byte Reload
	vmovupd	48(%rax,%rdx), %xmm13
	vmovupd	32(%rax,%rdx), %xmm7
	movq	1184(%rsp), %rdx        # 8-byte Reload
	vmovupd	48(%rax,%rdx), %xmm15
	vmovupd	32(%rax,%rdx), %xmm10
	movq	1152(%rsp), %rdx        # 8-byte Reload
	vmovupd	48(%rax,%rdx), %xmm12
	vmovupd	32(%rax,%rdx), %xmm14
	movq	1120(%rsp), %rdx        # 8-byte Reload
	vmovupd	48(%rax,%rdx), %xmm0
	vmovupd	32(%rax,%rdx), %xmm1
	movq	1088(%rsp), %rdx        # 8-byte Reload
	vmovupd	48(%rax,%rdx), %xmm2
	vmovupd	32(%rax,%rdx), %xmm3
	vmovupd	128(%rsp), %ymm8        # 32-byte Folded Reload
	vmaskmovpd	%ymm8, %ymm6, (%r9,%rcx)
	vinsertf128	$1, %xmm2, %ymm3, %ymm2
	vinsertf128	$1, %xmm0, %ymm1, %ymm0
	vaddpd	%ymm2, %ymm0, %ymm1
	vinsertf128	$1, %xmm12, %ymm14, %ymm0
	vinsertf128	$1, %xmm15, %ymm10, %ymm2
	vaddpd	%ymm0, %ymm2, %ymm0
	vinsertf128	$1, %xmm13, %ymm7, %ymm2
	vinsertf128	$1, %xmm5, %ymm11, %ymm3
	vaddpd	%ymm3, %ymm0, %ymm5
	vaddpd	%ymm2, %ymm1, %ymm0
	vinsertf128	$1, %xmm4, %ymm9, %ymm1
	vaddpd	%ymm1, %ymm0, %ymm0
	vmovupd	864(%rsp), %ymm1        # 32-byte Folded Reload
	vinsertf128	$1, 80(%rsp), %ymm1, %ymm1 # 16-byte Folded Reload
	vmovupd	160(%rsp), %ymm2        # 32-byte Folded Reload
	vinsertf128	$1, 208(%rsp), %ymm2, %ymm2 # 16-byte Folded Reload
	vaddpd	%ymm2, %ymm0, %ymm0
	vaddpd	%ymm1, %ymm5, %ymm1
	vmovupd	224(%rsp), %ymm2        # 32-byte Folded Reload
	vinsertf128	$1, 272(%rsp), %ymm2, %ymm2 # 16-byte Folded Reload
	vaddpd	%ymm2, %ymm1, %ymm1
	vmovupd	288(%rsp), %ymm2        # 32-byte Folded Reload
	vinsertf128	$1, 336(%rsp), %ymm2, %ymm2 # 16-byte Folded Reload
	vmovupd	352(%rsp), %ymm3        # 32-byte Folded Reload
	vinsertf128	$1, 400(%rsp), %ymm3, %ymm3 # 16-byte Folded Reload
	vaddpd	%ymm3, %ymm1, %ymm1
	vaddpd	%ymm2, %ymm0, %ymm2
	vmovupd	416(%rsp), %ymm0        # 32-byte Folded Reload
	vinsertf128	$1, 464(%rsp), %ymm0, %ymm0 # 16-byte Folded Reload
	vmovupd	544(%rsp), %ymm3        # 32-byte Folded Reload
	vinsertf128	$1, 592(%rsp), %ymm3, %ymm3 # 16-byte Folded Reload
	vaddpd	%ymm0, %ymm3, %ymm0
	vmovupd	640(%rsp), %ymm3        # 32-byte Folded Reload
	vinsertf128	$1, 704(%rsp), %ymm3, %ymm3 # 16-byte Folded Reload
	vaddpd	%ymm3, %ymm0, %ymm0
	vmovupd	1344(%rsp), %ymm3       # 32-byte Folded Reload
	vmulpd	%ymm2, %ymm3, %ymm2
	vmovupd	-64(%rsp), %ymm3        # 32-byte Folded Reload
	vmulpd	%ymm1, %ymm3, %ymm1
	vmovapd	%ymm3, %ymm14
	vmovupd	736(%rsp), %ymm3        # 32-byte Folded Reload
	vinsertf128	$1, 768(%rsp), %ymm3, %ymm3 # 16-byte Folded Reload
	vmovupd	480(%rsp), %ymm4        # 32-byte Folded Reload
	vinsertf128	$1, 528(%rsp), %ymm4, %ymm4 # 16-byte Folded Reload
	vmovupd	-32(%rsp), %ymm5        # 32-byte Folded Reload
	vmulpd	%ymm4, %ymm5, %ymm4
	vmovapd	%ymm5, %ymm15
	vaddpd	%ymm4, %ymm1, %ymm1
	vmovapd	.LCPI0_2(%rip), %ymm5
	vmovupd	608(%rsp), %ymm4        # 32-byte Folded Reload
	vmulpd	%ymm5, %ymm4, %ymm4
	vmovapd	%ymm5, %ymm13
	vaddpd	%ymm1, %ymm2, %ymm2
	vsubpd	672(%rsp), %ymm4, %ymm1 # 32-byte Folded Reload
	vaddpd	%ymm3, %ymm0, %ymm0
	vmovupd	800(%rsp), %ymm3        # 32-byte Folded Reload
	vinsertf128	$1, 832(%rsp), %ymm3, %ymm3 # 16-byte Folded Reload
	vaddpd	%ymm3, %ymm0, %ymm0
	vmovupd	896(%rsp), %ymm3        # 32-byte Folded Reload
	vinsertf128	$1, 960(%rsp), %ymm3, %ymm3 # 16-byte Folded Reload
	vaddpd	%ymm3, %ymm0, %ymm0
	vmovupd	1312(%rsp), %ymm3       # 32-byte Folded Reload
	vmulpd	%ymm0, %ymm3, %ymm0
	vaddpd	%ymm0, %ymm2, %ymm0
	vmulpd	992(%rsp), %ymm0, %ymm0 # 32-byte Folded Reload
	vaddpd	%ymm0, %ymm1, %ymm0
	vmovupd	96(%rsp), %ymm1         # 32-byte Folded Reload
	vmaskmovpd	%ymm0, %ymm1, 32(%r9,%rcx)
.LBB0_14:                               # %safe_if_after_true.us
                                        #   in Loop: Header=BB0_12 Depth=3
	addl	$64, %r13d
	movl	1216(%rsp), %ecx        # 4-byte Reload
	addl	$8, %ecx
	cmpl	1308(%rsp), %ecx        # 4-byte Folded Reload
	jl	.LBB0_12
# BB#15:                                # %for_exit40.us
                                        #   in Loop: Header=BB0_16 Depth=2
	movl	-80(%rsp), %edx         # 4-byte Reload
	addl	-76(%rsp), %edx         # 4-byte Folded Reload
	movl	-84(%rsp), %edi         # 4-byte Reload
	incl	%edi
	cmpl	-72(%rsp), %edi         # 4-byte Folded Reload
	movl	-92(%rsp), %ecx         # 4-byte Reload
	movl	-88(%rsp), %esi         # 4-byte Reload
	jne	.LBB0_16
.LBB0_11:                               # %for_exit33
                                        #   in Loop: Header=BB0_9 Depth=1
	movl	-100(%rsp), %edx        # 4-byte Reload
	addl	-104(%rsp), %edx        # 4-byte Folded Reload
	movl	%edx, -100(%rsp)        # 4-byte Spill
	incl	%esi
	cmpl	%ecx, %esi
	jne	.LBB0_9
	jmp	.LBB0_6
.LBB0_1:                                # %for_test264.preheader
	cmpl	%r9d, %r8d
	jge	.LBB0_6
# BB#2:                                 # %for_test275.preheader.lr.ph
	leal	2(%r8), %r13d
	movl	%esi, %r10d
	imull	%r10d, %r13d
	movl	%r10d, %ecx
	imull	%r8d, %ecx
	movl	%edx, %esi
	movl	%esi, -96(%rsp)         # 4-byte Spill
	leal	(%rsi,%rcx), %r15d
	movl	%r9d, -92(%rsp)         # 4-byte Spill
	leal	2(%rsi,%rcx), %edx
	movl	%edx, 1248(%rsp)        # 4-byte Spill
	leal	-1(%rsi,%rcx), %edx
	movl	%edx, 1344(%rsp)        # 4-byte Spill
	leal	3(%rsi,%rcx), %r12d
	leal	-2(%rsi,%rcx), %edx
	movl	%edx, 1312(%rsp)        # 4-byte Spill
	leal	-3(%rsi,%rcx), %edi
	addl	%esi, %r13d
	leal	1(%rsi,%rcx), %ecx
	leal	-3(%r8), %r14d
	imull	%r10d, %r14d
	leal	-2(%r8), %r9d
	imull	%r10d, %r9d
	leal	3(%r8), %ebx
	imull	%r10d, %ebx
	leal	-1(%r8), %ebp
	imull	%r10d, %ebp
	leal	1(%r8), %edx
	imull	%r10d, %edx
	addl	%esi, %edx
	addl	%esi, %ebp
	addl	%esi, %ebx
	addl	%esi, %r9d
	addl	%esi, %r14d
	vmovd	1308(%rsp), %xmm5       # 4-byte Folded Reload
	movl	1440(%rsp), %r11d
	imull	%r11d, %ecx
	movl	%ecx, 1184(%rsp)        # 4-byte Spill
	imull	%r11d, %r13d
	imull	%r11d, %edi
	movl	%edi, 1216(%rsp)        # 4-byte Spill
	movl	1312(%rsp), %ecx        # 4-byte Reload
	imull	%r11d, %ecx
	movl	%ecx, 1312(%rsp)        # 4-byte Spill
	imull	%r11d, %r12d
	movl	1344(%rsp), %esi        # 4-byte Reload
	imull	%r11d, %esi
	movl	%esi, 1344(%rsp)        # 4-byte Spill
	movl	1248(%rsp), %ecx        # 4-byte Reload
	imull	%r11d, %ecx
	imull	%r11d, %r15d
	movl	-68(%rsp), %esi         # 4-byte Reload
	leal	(,%rsi,8), %esi
	imull	%r11d, %r14d
	imull	%r11d, %r9d
	imull	%r11d, %ebx
	imull	%r11d, %ebp
	imull	%r11d, %edx
	leal	-16(%rsi,%r15,8), %edi
	movl	%edi, 672(%rsp)         # 4-byte Spill
	leal	(%rsi,%r15,8), %edi
	movl	%edi, 640(%rsp)         # 4-byte Spill
	movl	1184(%rsp), %edi        # 4-byte Reload
	leal	(%rsi,%rdi,8), %edi
	movl	%edi, 608(%rsp)         # 4-byte Spill
	movl	%r8d, %edi
	leal	(%rsi,%rcx,8), %ecx
	movl	%ecx, 592(%rsp)         # 4-byte Spill
	movl	1344(%rsp), %ecx        # 4-byte Reload
	leal	(%rsi,%rcx,8), %ecx
	movl	%ecx, 544(%rsp)         # 4-byte Spill
	leal	(%rsi,%r12,8), %ecx
	movl	%ecx, 528(%rsp)         # 4-byte Spill
	movl	1312(%rsp), %ecx        # 4-byte Reload
	leal	(%rsi,%rcx,8), %ecx
	movl	%ecx, 480(%rsp)         # 4-byte Spill
	movl	1216(%rsp), %ecx        # 4-byte Reload
	leal	(%rsi,%rcx,8), %ecx
	movl	%ecx, 464(%rsp)         # 4-byte Spill
	leal	(%rsi,%rdx,8), %ecx
	movl	%ecx, 416(%rsp)         # 4-byte Spill
	leal	(%rsi,%r13,8), %ecx
	movl	%ecx, 400(%rsp)         # 4-byte Spill
	leal	(%rsi,%rbp,8), %ecx
	movl	%ecx, 352(%rsp)         # 4-byte Spill
	leal	(%rsi,%rbx,8), %ecx
	movl	%ecx, 336(%rsp)         # 4-byte Spill
	leal	(%rsi,%r9,8), %ecx
	movl	%ecx, 288(%rsp)         # 4-byte Spill
	leal	(%rsi,%r14,8), %ecx
	movl	%ecx, 272(%rsp)         # 4-byte Spill
	movl	$0, 160(%rsp)           # 4-byte Folded Spill
	imull	%r11d, %r10d
	shll	$3, %r11d
	movl	%r11d, -76(%rsp)        # 4-byte Spill
	shll	$3, %r10d
	movl	%r10d, -104(%rsp)       # 4-byte Spill
	vpermilpd	$0, %xmm1, %xmm6 # xmm6 = xmm1[0,0]
	vpermilpd	$0, %xmm3, %xmm3 # xmm3 = xmm3[0,0]
	vpermilpd	$0, %xmm2, %xmm1 # xmm1 = xmm2[0,0]
	vmovaps	%ymm0, %ymm8
	vmovups	%ymm8, 704(%rsp)        # 32-byte Folded Spill
	vextractf128	$1, %ymm8, %xmm7
	vpshufd	$80, %xmm8, %xmm0       # xmm0 = xmm8[0,0,1,1]
	vinsertf128	$1, %xmm6, %ymm6, %ymm13
	vpshufd	$80, %xmm7, %xmm2       # xmm2 = xmm7[0,0,1,1]
	vinsertf128	$1, %xmm3, %ymm3, %ymm15
	vpshufd	$-6, %xmm7, %xmm3       # xmm3 = xmm7[2,2,3,3]
	vinsertf128	$1, %xmm1, %ymm1, %ymm10
	vpshufd	$-6, %xmm8, %xmm1       # xmm1 = xmm8[2,2,3,3]
	vpshufd	$0, %xmm5, %xmm7        # xmm7 = xmm5[0,0,0,0]
	vpermilpd	$0, %xmm4, %xmm4 # xmm4 = xmm4[0,0]
	vinsertf128	$1, %xmm4, %ymm4, %ymm4
	vmovupd	%ymm4, 1344(%rsp)       # 32-byte Folded Spill
	vinsertf128	$1, %xmm3, %ymm2, %ymm5
	vinsertf128	$1, %xmm1, %ymm0, %ymm6
	vinsertf128	$1, %xmm7, %ymm7, %ymm0
	vmovups	%ymm0, 1312(%rsp)       # 32-byte Folded Spill
	vmovapd	.LCPI0_2(%rip), %ymm14
	.align	16, 0x90
.LBB0_3:                                # %for_test275.preheader
                                        # =>This Loop Header: Depth=1
                                        #     Child Loop BB0_21 Depth 2
                                        #       Child Loop BB0_17 Depth 3
	movl	%edi, -88(%rsp)         # 4-byte Spill
	movl	-96(%rsp), %ecx         # 4-byte Reload
	cmpl	-72(%rsp), %ecx         # 4-byte Folded Reload
	jge	.LBB0_5
# BB#4:                                 # %for_test286.preheader.lr.ph
                                        #   in Loop: Header=BB0_3 Depth=1
	movl	-68(%rsp), %ecx         # 4-byte Reload
	cmpl	1308(%rsp), %ecx        # 4-byte Folded Reload
	movl	160(%rsp), %ecx         # 4-byte Reload
	movl	-96(%rsp), %edx         # 4-byte Reload
	jge	.LBB0_5
	.align	16, 0x90
.LBB0_21:                               # %for_loop288.lr.ph.us
                                        #   Parent Loop BB0_3 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB0_17 Depth 3
	movl	%edx, 208(%rsp)         # 4-byte Spill
	movl	%ecx, 224(%rsp)         # 4-byte Spill
	movl	%ecx, %r9d
	movl	-68(%rsp), %r15d        # 4-byte Reload
	.align	16, 0x90
.LBB0_17:                               # %for_loop288.us
                                        #   Parent Loop BB0_3 Depth=1
                                        #     Parent Loop BB0_21 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	vmovups	1312(%rsp), %ymm3       # 32-byte Folded Reload
	vmovups	%ymm3, 1312(%rsp)       # 32-byte Folded Spill
	vextractf128	$1, %ymm3, %xmm0
	vmovd	%r15d, %xmm1
	vpshufd	$0, %xmm1, %xmm1        # xmm1 = xmm1[0,0,0,0]
	vpaddd	.LCPI0_0(%rip), %xmm1, %xmm2
	vpcmpgtd	%xmm2, %xmm0, %xmm0
	vpaddd	.LCPI0_1(%rip), %xmm1, %xmm1
	vpcmpgtd	%xmm1, %xmm3, %xmm1
	vinsertf128	$1, %xmm0, %ymm1, %ymm0
	vandps	704(%rsp), %ymm0, %ymm11 # 32-byte Folded Reload
	vmovmskps	%ymm11, %ecx
	testl	%ecx, %ecx
	je	.LBB0_19
# BB#18:                                # %safe_if_run_true467.us
                                        #   in Loop: Header=BB0_17 Depth=3
	movl	640(%rsp), %r11d        # 4-byte Reload
	leal	24(%r11,%r9), %ecx
	movl	528(%rsp), %edx         # 4-byte Reload
	leal	(%rdx,%r9), %ebx
	leal	8(%r11,%r9), %edx
	movl	%edx, 1088(%rsp)        # 4-byte Spill
	movl	608(%rsp), %edx         # 4-byte Reload
	leal	(%rdx,%r9), %edx
	movl	%edx, 1048(%rsp)        # 4-byte Spill
	movl	480(%rsp), %edx         # 4-byte Reload
	leal	(%rdx,%r9), %edx
	movl	%edx, 768(%rsp)         # 4-byte Spill
	movl	464(%rsp), %edx         # 4-byte Reload
	leal	(%rdx,%r9), %r14d
	movl	592(%rsp), %esi         # 4-byte Reload
	leal	(%rsi,%r9), %esi
	movl	672(%rsp), %edi         # 4-byte Reload
	leal	(%rdi,%r9), %ebp
	leal	16(%r11,%r9), %r12d
	leal	-8(%rdi,%r9), %r13d
	movl	336(%rsp), %edx         # 4-byte Reload
	leal	(%rdx,%r9), %edx
	movl	%edx, 832(%rsp)         # 4-byte Spill
	movl	272(%rsp), %edx         # 4-byte Reload
	leal	(%rdx,%r9), %edx
	movl	%edx, 800(%rsp)         # 4-byte Spill
	leal	8(%rdi,%r9), %r10d
	movl	544(%rsp), %r8d         # 4-byte Reload
	leal	(%r8,%r9), %edx
	movl	%edx, 960(%rsp)         # 4-byte Spill
	leal	(%r11,%r9), %edx
	movl	%edx, 928(%rsp)         # 4-byte Spill
	movl	416(%rsp), %edi         # 4-byte Reload
	leal	(%rdi,%r9), %edx
	movl	%edx, 896(%rsp)         # 4-byte Spill
	movl	400(%rsp), %edi         # 4-byte Reload
	leal	(%rdi,%r9), %edx
	movl	%edx, 864(%rsp)         # 4-byte Spill
	movl	288(%rsp), %edx         # 4-byte Reload
	leal	(%rdx,%r9), %edx
	movl	%edx, 992(%rsp)         # 4-byte Spill
	movl	352(%rsp), %edi         # 4-byte Reload
	leal	(%rdi,%r9), %r8d
	movslq	%ecx, %rcx
	movq	%rcx, 1184(%rsp)        # 8-byte Spill
	vmaskmovpd	(%rax,%rcx), %ymm6, %ymm0
	movslq	%r13d, %rcx
	movq	%rcx, 1152(%rsp)        # 8-byte Spill
	vmaskmovpd	(%rax,%rcx), %ymm6, %ymm1
	vaddpd	%ymm0, %ymm1, %ymm0
	movslq	%r12d, %rcx
	movq	%rcx, 1248(%rsp)        # 8-byte Spill
	movslq	%ebx, %rdx
	movq	%rdx, 1120(%rsp)        # 8-byte Spill
	vmaskmovpd	(%rax,%rdx), %ymm6, %ymm1
	vaddpd	%ymm1, %ymm0, %ymm0
	vmaskmovpd	(%rax,%rcx), %ymm6, %ymm1
	movslq	%ebp, %rcx
	movq	%rcx, 1216(%rsp)        # 8-byte Spill
	vmaskmovpd	(%rax,%rcx), %ymm6, %ymm2
	vaddpd	%ymm1, %ymm2, %ymm1
	movslq	%esi, %rsi
	movq	%rsi, 1056(%rsp)        # 8-byte Spill
	movslq	%r14d, %rdx
	vmaskmovpd	(%rax,%rdx), %ymm6, %ymm2
	vaddpd	%ymm2, %ymm0, %ymm0
	movslq	768(%rsp), %rcx         # 4-byte Folded Reload
	movslq	1048(%rsp), %rdi        # 4-byte Folded Reload
	movq	%rdi, 1048(%rsp)        # 8-byte Spill
	vmaskmovpd	(%rax,%rsi), %ymm6, %ymm2
	movslq	1088(%rsp), %rsi        # 4-byte Folded Reload
	movq	%rsi, 1088(%rsp)        # 8-byte Spill
	vmaskmovpd	(%rax,%rsi), %ymm6, %ymm3
	movslq	%r10d, %r11
	vmaskmovpd	(%rax,%r11), %ymm6, %ymm4
	vaddpd	%ymm3, %ymm4, %ymm3
	vaddpd	%ymm2, %ymm1, %ymm1
	movslq	800(%rsp), %rsi         # 4-byte Folded Reload
	vmaskmovpd	(%rax,%rdi), %ymm6, %ymm7
	vmaskmovpd	(%rax,%rcx), %ymm6, %ymm2
	movslq	832(%rsp), %rdi         # 4-byte Folded Reload
	vmaskmovpd	(%rax,%rdi), %ymm6, %ymm8
	vpshufd	$80, %xmm11, %xmm4      # xmm4 = xmm11[0,0,1,1]
	vaddpd	%ymm8, %ymm0, %ymm0
	vaddpd	%ymm2, %ymm1, %ymm2
	vaddpd	%ymm7, %ymm3, %ymm3
	vmaskmovpd	(%rax,%rsi), %ymm6, %ymm1
	movslq	864(%rsp), %r12         # 4-byte Folded Reload
	movslq	896(%rsp), %rbx         # 4-byte Folded Reload
	vpshufd	$-6, %xmm11, %xmm7      # xmm7 = xmm11[2,2,3,3]
	vinsertf128	$1, %xmm7, %ymm4, %ymm12
	movslq	928(%rsp), %r13         # 4-byte Folded Reload
	movslq	960(%rsp), %r10         # 4-byte Folded Reload
	vmaskmovpd	(%rax,%r10), %ymm6, %ymm4
	vaddpd	%ymm4, %ymm3, %ymm4
	vmaskmovpd	(%rax,%r13), %ymm12, %ymm7
	vmaskmovpd	(%rax,%rbx), %ymm6, %ymm8
	vextractf128	$1, %ymm11, %xmm3
	vmaskmovpd	(%rax,%r12), %ymm6, %ymm9
	vaddpd	%ymm9, %ymm2, %ymm2
	movslq	992(%rsp), %rbp         # 4-byte Folded Reload
	vmaskmovpd	(%rax,%rbp), %ymm6, %ymm9
	vaddpd	%ymm9, %ymm2, %ymm2
	vaddpd	%ymm1, %ymm0, %ymm1
	vmulpd	%ymm14, %ymm7, %ymm0
	vaddpd	%ymm8, %ymm4, %ymm4
	vmaskmovpd	(%rax,%r13), %ymm6, %ymm7
	movslq	%r8d, %r8
	vmaskmovpd	(%rax,%r8), %ymm6, %ymm8
	vaddpd	%ymm8, %ymm4, %ymm8
	vmovapd	%ymm10, %ymm14
	vmulpd	%ymm7, %ymm14, %ymm7
	vpshufd	$-6, %xmm3, %xmm4       # xmm4 = xmm3[2,2,3,3]
	vpshufd	$80, %xmm3, %xmm3       # xmm3 = xmm3[0,0,1,1]
	movq	1480(%rsp), %r14
	vmaskmovpd	(%r14,%r13), %ymm12, %ymm9
	vsubpd	%ymm9, %ymm0, %ymm0
	vmulpd	%ymm1, %ymm13, %ymm1
	vmulpd	%ymm2, %ymm15, %ymm2
	vmovupd	1344(%rsp), %ymm9       # 32-byte Folded Reload
	vmovupd	%ymm9, 1344(%rsp)       # 32-byte Folded Spill
	vmulpd	%ymm8, %ymm9, %ymm8
	vaddpd	%ymm7, %ymm8, %ymm7
	vmaskmovpd	32(%rax,%rsi), %ymm5, %ymm8
	vmovupd	%ymm8, 992(%rsp)        # 32-byte Folded Spill
	vinsertf128	$1, %xmm4, %ymm3, %ymm11
	vmaskmovpd	32(%rax,%rdi), %ymm5, %ymm3
	vmovupd	%ymm3, 960(%rsp)        # 32-byte Folded Spill
	vaddpd	%ymm7, %ymm2, %ymm2
	vmaskmovpd	32(%rax,%rdx), %ymm5, %ymm3
	vmovupd	%ymm3, 928(%rsp)        # 32-byte Folded Spill
	vaddpd	%ymm1, %ymm2, %ymm1
	movq	1464(%rsp), %rdx
	vmaskmovpd	(%rdx,%r13), %ymm12, %ymm2
	vmulpd	%ymm2, %ymm1, %ymm1
	movq	1120(%rsp), %rsi        # 8-byte Reload
	vmaskmovpd	32(%rax,%rsi), %ymm5, %ymm2
	vmovupd	%ymm2, 1120(%rsp)       # 32-byte Folded Spill
	vaddpd	%ymm1, %ymm0, %ymm0
	vmovupd	%ymm0, 736(%rsp)        # 32-byte Folded Spill
	vmaskmovpd	32(%rax,%r8), %ymm5, %ymm0
	vmovupd	%ymm0, 896(%rsp)        # 32-byte Folded Spill
	movq	1184(%rsp), %rsi        # 8-byte Reload
	vmaskmovpd	32(%rax,%rsi), %ymm5, %ymm0
	vmovupd	%ymm0, 1184(%rsp)       # 32-byte Folded Spill
	movq	1152(%rsp), %rsi        # 8-byte Reload
	vmaskmovpd	32(%rax,%rsi), %ymm5, %ymm0
	vmovupd	%ymm0, 1152(%rsp)       # 32-byte Folded Spill
	vmaskmovpd	32(%rax,%rbp), %ymm5, %ymm0
	vmovupd	%ymm0, 832(%rsp)        # 32-byte Folded Spill
	vmaskmovpd	32(%rax,%r12), %ymm5, %ymm0
	vmovupd	%ymm0, 800(%rsp)        # 32-byte Folded Spill
	vmaskmovpd	32(%rax,%rcx), %ymm5, %ymm0
	vmovupd	%ymm0, 768(%rsp)        # 32-byte Folded Spill
	vmaskmovpd	32(%rax,%r13), %ymm5, %ymm0
	vmovupd	%ymm0, 864(%rsp)        # 32-byte Folded Spill
	movq	1056(%rsp), %rcx        # 8-byte Reload
	vmaskmovpd	32(%rax,%rcx), %ymm5, %ymm0
	vmovupd	%ymm0, 1056(%rsp)       # 32-byte Folded Spill
	vmaskmovpd	32(%rax,%rbx), %ymm5, %ymm7
	vmaskmovpd	32(%rax,%r10), %ymm5, %ymm10
	movq	1048(%rsp), %rcx        # 8-byte Reload
	vmaskmovpd	32(%rax,%rcx), %ymm5, %ymm0
	movq	1088(%rsp), %rcx        # 8-byte Reload
	vmaskmovpd	32(%rax,%rcx), %ymm5, %ymm1
	vmaskmovpd	32(%rax,%r11), %ymm5, %ymm2
	movq	1248(%rsp), %rcx        # 8-byte Reload
	vmaskmovpd	32(%rax,%rcx), %ymm5, %ymm3
	movq	1216(%rsp), %rcx        # 8-byte Reload
	vmaskmovpd	32(%rax,%rcx), %ymm5, %ymm4
	vmaskmovpd	32(%rdx,%r13), %ymm11, %ymm8
	vmovupd	%ymm8, 1248(%rsp)       # 32-byte Folded Spill
	vmaskmovpd	32(%r14,%r13), %ymm11, %ymm8
	vmovupd	%ymm8, 1216(%rsp)       # 32-byte Folded Spill
	vmaskmovpd	32(%rax,%r13), %ymm11, %ymm8
	vmovupd	%ymm8, 1088(%rsp)       # 32-byte Folded Spill
	vmovupd	736(%rsp), %ymm8        # 32-byte Folded Reload
	vmaskmovpd	%ymm8, %ymm12, (%r14,%r13)
	vaddpd	%ymm3, %ymm4, %ymm3
	vaddpd	%ymm1, %ymm2, %ymm1
	vaddpd	%ymm0, %ymm1, %ymm0
	vaddpd	%ymm10, %ymm0, %ymm0
	vaddpd	%ymm7, %ymm0, %ymm1
	vaddpd	1056(%rsp), %ymm3, %ymm0 # 32-byte Folded Reload
	vaddpd	768(%rsp), %ymm0, %ymm0 # 32-byte Folded Reload
	vaddpd	800(%rsp), %ymm0, %ymm0 # 32-byte Folded Reload
	vaddpd	832(%rsp), %ymm0, %ymm2 # 32-byte Folded Reload
	vmovupd	1152(%rsp), %ymm0       # 32-byte Folded Reload
	vaddpd	1184(%rsp), %ymm0, %ymm0 # 32-byte Folded Reload
	vmulpd	%ymm2, %ymm15, %ymm2
	vaddpd	896(%rsp), %ymm1, %ymm1 # 32-byte Folded Reload
	vmulpd	%ymm1, %ymm9, %ymm1
	vmulpd	864(%rsp), %ymm14, %ymm3 # 32-byte Folded Reload
	vmovapd	%ymm14, %ymm10
	vaddpd	%ymm3, %ymm1, %ymm3
	vmovapd	.LCPI0_2(%rip), %ymm4
	vmovupd	1088(%rsp), %ymm1       # 32-byte Folded Reload
	vmulpd	%ymm4, %ymm1, %ymm1
	vmovapd	%ymm4, %ymm14
	vsubpd	1216(%rsp), %ymm1, %ymm1 # 32-byte Folded Reload
	vaddpd	%ymm3, %ymm2, %ymm2
	vaddpd	1120(%rsp), %ymm0, %ymm0 # 32-byte Folded Reload
	vaddpd	928(%rsp), %ymm0, %ymm0 # 32-byte Folded Reload
	vaddpd	960(%rsp), %ymm0, %ymm0 # 32-byte Folded Reload
	vaddpd	992(%rsp), %ymm0, %ymm0 # 32-byte Folded Reload
	vmulpd	%ymm0, %ymm13, %ymm0
	vaddpd	%ymm0, %ymm2, %ymm0
	vmulpd	1248(%rsp), %ymm0, %ymm0 # 32-byte Folded Reload
	vaddpd	%ymm0, %ymm1, %ymm0
	vmaskmovpd	%ymm0, %ymm11, 32(%r14,%r13)
.LBB0_19:                               # %safe_if_after_true466.us
                                        #   in Loop: Header=BB0_17 Depth=3
	addl	$64, %r9d
	addl	$8, %r15d
	cmpl	1308(%rsp), %r15d       # 4-byte Folded Reload
	jl	.LBB0_17
# BB#20:                                # %for_exit289.us
                                        #   in Loop: Header=BB0_21 Depth=2
	movl	224(%rsp), %ecx         # 4-byte Reload
	addl	-76(%rsp), %ecx         # 4-byte Folded Reload
	movl	208(%rsp), %edx         # 4-byte Reload
	incl	%edx
	cmpl	-72(%rsp), %edx         # 4-byte Folded Reload
	jne	.LBB0_21
.LBB0_5:                                # %for_exit278
                                        #   in Loop: Header=BB0_3 Depth=1
	movl	160(%rsp), %ecx         # 4-byte Reload
	addl	-104(%rsp), %ecx        # 4-byte Folded Reload
	movl	%ecx, 160(%rsp)         # 4-byte Spill
	movl	-88(%rsp), %edi         # 4-byte Reload
	incl	%edi
	movl	-92(%rsp), %ecx         # 4-byte Reload
	cmpl	%ecx, %edi
	jne	.LBB0_3
.LBB0_6:                                # %for_exit
	addq	$1384, %rsp             # imm = 0x568
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	vzeroupper
	ret
.Ltmp0:
	.size	stencil_step___uniuniuniuniuniuniuniuniuniun_3C_Cund_3E_un_3C_Cund_3E_un_3C_Cund_3E_un_3C_und_3E_, .Ltmp0-stencil_step___uniuniuniuniuniuniuniuniuniun_3C_Cund_3E_un_3C_Cund_3E_un_3C_Cund_3E_un_3C_und_3E_

	.align	16, 0x90
	.type	stencil_step_task___uniuniuniuniuniuniuniuniun_3C_Cund_3E_un_3C_Cund_3E_un_3C_Cund_3E_un_3C_und_3E_,@function
stencil_step_task___uniuniuniuniuniuniuniuniun_3C_Cund_3E_un_3C_Cund_3E_un_3C_Cund_3E_un_3C_und_3E_: # @stencil_step_task___uniuniuniuniuniuniuniuniun_3C_Cund_3E_un_3C_Cund_3E_un_3C_Cund_3E_un_3C_und_3E_
# BB#0:                                 # %allocas
	pushq	%rbp
	pushq	%r15
	pushq	%r14
	pushq	%rbx
	subq	$56, %rsp
	movq	%rdi, %rax
	movl	16(%rax), %r8d
	movq	56(%rax), %rbx
	movq	48(%rax), %r15
	movq	40(%rax), %r14
	movq	32(%rax), %r11
	leal	1(%r8,%rcx), %r9d
	movl	24(%rax), %r10d
	vmovaps	64(%rax), %ymm0
	addl	%ecx, %r8d
	movl	20(%rax), %ebp
	movl	12(%rax), %ecx
	movl	8(%rax), %edx
	movl	(%rax), %edi
	movl	4(%rax), %esi
	vmovmskps	%ymm0, %eax
	cmpl	$255, %eax
	jne	.LBB1_2
# BB#1:                                 # %all_on
	vpcmpeqd	%xmm0, %xmm0, %xmm0
	movq	%rbx, 40(%rsp)
	movq	%r15, 32(%rsp)
	movq	%r14, 24(%rsp)
	movq	%r11, 16(%rsp)
	movl	%r10d, 8(%rsp)
	movl	%ebp, (%rsp)
	vinsertf128	$1, %xmm0, %ymm0, %ymm0
	jmp	.LBB1_3
.LBB1_2:                                # %some_on
	movq	%rbx, 40(%rsp)
	movq	%r15, 32(%rsp)
	movq	%r14, 24(%rsp)
	movq	%r11, 16(%rsp)
	movl	%r10d, 8(%rsp)
	movl	%ebp, (%rsp)
.LBB1_3:                                # %some_on
	callq	stencil_step___uniuniuniuniuniuniuniuniuniun_3C_Cund_3E_un_3C_Cund_3E_un_3C_Cund_3E_un_3C_und_3E_
	addq	$56, %rsp
	popq	%rbx
	popq	%r14
	popq	%r15
	popq	%rbp
	vzeroupper
	ret
.Ltmp1:
	.size	stencil_step_task___uniuniuniuniuniuniuniuniun_3C_Cund_3E_un_3C_Cund_3E_un_3C_Cund_3E_un_3C_und_3E_, .Ltmp1-stencil_step_task___uniuniuniuniuniuniuniuniun_3C_Cund_3E_un_3C_Cund_3E_un_3C_Cund_3E_un_3C_und_3E_

	.globl	loop_stencil_ispc_tasks
	.align	16, 0x90
	.type	loop_stencil_ispc_tasks,@function
loop_stencil_ispc_tasks:                # @loop_stencil_ispc_tasks
# BB#0:                                 # %allocas
	pushq	%rbp
	pushq	%r15
	pushq	%r14
	pushq	%r13
	pushq	%r12
	pushq	%rbx
	subq	$104, %rsp
	movl	%r9d, 92(%rsp)          # 4-byte Spill
	movl	%r8d, 88(%rsp)          # 4-byte Spill
	movl	%ecx, 84(%rsp)          # 4-byte Spill
	movl	%edx, 80(%rsp)          # 4-byte Spill
	movl	%esi, %ebx
	movl	%edi, %ebp
	movq	$0, 96(%rsp)
	cmpl	%ebx, %ebp
	jge	.LBB2_10
# BB#1:                                 # %for_loop.lr.ph
	movq	216(%rsp), %r13
	movl	168(%rsp), %r14d
	movl	160(%rsp), %r12d
	subl	%r12d, %r14d
	leaq	96(%rsp), %r15
	vpcmpeqd	%xmm0, %xmm0, %xmm0
	vinsertf128	$1, %xmm0, %ymm0, %ymm1
	vmovups	%ymm1, 32(%rsp)         # 32-byte Folded Spill
	vinsertf128	$1, %xmm0, %ymm0, %ymm0
	vmovups	%ymm0, (%rsp)           # 32-byte Folded Spill
	.align	16, 0x90
.LBB2_2:                                # %for_loop
                                        # =>This Inner Loop Header: Depth=1
	movq	%r15, %rdi
	movl	$96, %esi
	movl	$32, %edx
	vzeroupper
	callq	ISPCAlloc
	movq	%rax, %rdx
	movl	80(%rsp), %eax          # 4-byte Reload
	movl	%eax, (%rdx)
	movl	84(%rsp), %eax          # 4-byte Reload
	movl	%eax, 4(%rdx)
	movl	88(%rsp), %eax          # 4-byte Reload
	movl	%eax, 8(%rdx)
	movl	92(%rsp), %eax          # 4-byte Reload
	movl	%eax, 12(%rdx)
	movl	%r12d, 16(%rdx)
	movl	176(%rsp), %eax
	movl	%eax, 20(%rdx)
	movl	184(%rsp), %eax
	movl	%eax, 24(%rdx)
	testb	$1, %bpl
	movl	192(%rsp), %eax
	movl	%eax, 28(%rdx)
	movq	200(%rsp), %rax
	movq	%rax, 32(%rdx)
	movq	208(%rsp), %rax
	movq	%rax, 40(%rdx)
	jne	.LBB2_4
# BB#3:                                 # %if_then
                                        #   in Loop: Header=BB2_2 Depth=1
	movq	%r13, 48(%rdx)
	movq	224(%rsp), %rax
	movq	%rax, 56(%rdx)
	vmovups	32(%rsp), %ymm0         # 32-byte Folded Reload
	jmp	.LBB2_5
	.align	16, 0x90
.LBB2_4:                                # %if_else
                                        #   in Loop: Header=BB2_2 Depth=1
	movq	224(%rsp), %rax
	movq	%rax, 48(%rdx)
	movq	%r13, 56(%rdx)
	vmovups	(%rsp), %ymm0           # 32-byte Folded Reload
.LBB2_5:                                # %if_else
                                        #   in Loop: Header=BB2_2 Depth=1
	vmovaps	%ymm0, 64(%rdx)
	movq	%r15, %rdi
	movl	$stencil_step_task___uniuniuniuniuniuniuniuniun_3C_Cund_3E_un_3C_Cund_3E_un_3C_Cund_3E_un_3C_und_3E_, %esi
	movl	%r14d, %ecx
	movl	$1, %r8d
	movl	$1, %r9d
	vzeroupper
	callq	ISPCLaunch
	movq	96(%rsp), %rdi
	testq	%rdi, %rdi
	je	.LBB2_7
# BB#6:                                 # %call_sync
                                        #   in Loop: Header=BB2_2 Depth=1
	callq	ISPCSync
	movq	$0, 96(%rsp)
.LBB2_7:                                # %post_sync
                                        #   in Loop: Header=BB2_2 Depth=1
	incl	%ebp
	cmpl	%ebp, %ebx
	jne	.LBB2_2
# BB#8:                                 # %for_exit
	movq	96(%rsp), %rdi
	testq	%rdi, %rdi
	je	.LBB2_10
# BB#9:                                 # %call_sync72
	callq	ISPCSync
	movq	$0, 96(%rsp)
.LBB2_10:                               # %post_sync73
	addq	$104, %rsp
	popq	%rbx
	popq	%r12
	popq	%r13
	popq	%r14
	popq	%r15
	popq	%rbp
	ret
.Ltmp2:
	.size	loop_stencil_ispc_tasks, .Ltmp2-loop_stencil_ispc_tasks


	.section	".note.GNU-stack","",@progbits
