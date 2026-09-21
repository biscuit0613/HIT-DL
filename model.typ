// 深度学习实验报告 Typst 模板
// 使用前只需修改下面四项信息，并将各节中的占位内容替换为自己的实验内容。

#let experiment-number = "1"
#let experiment-title = "在此填写实验题目"
#let student-id = "在此填写学号"
#let student-name = "在此填写姓名"

#set document(
  title: [《深度学习》实验报告],
  author: student-name,
)
#set page(
  paper: "a4",
  margin: (top: 2.5cm, bottom: 2.5cm, left: 2.8cm, right: 2.8cm),
  numbering: "1",
)
#set text(
  font: ("Noto Serif CJK SC", "Noto Sans CJK SC"),
  size: 11pt,
  lang: "zh",
)
#set par(
  justify: true,
  leading: 0.65em,
)
#set heading(numbering: none)

#align(center)[
  #v(3.6cm)
  #text(size: 25pt, weight: "bold")[《深度学习》实验报告]
  #v(1.5cm)
  #text(size: 17pt)[（第 #experiment-number 次）]
  #v(3.2cm)
  #table(
    columns: (3.2cm, 9cm),
    align: left,
    inset: 8pt,
    stroke: none,
    [实验题目], [#experiment-title],
    [学号], [#student-id],
    [姓名], [#student-name],
  )
]

#pagebreak()

= 实验报告内容

= 1. 实验目的

在此说明本次实验需要掌握的理论知识、实践技能和待解决的问题。应结合具体实验任务，明确实验目标。

= 2. 实验内容

在此概述实验任务、数据集、模型、主要步骤和需要完成的功能。若实验包含多个方案或模块，请说明本次实际选择的方案。

= 3. 实验环境

在此列出硬件环境、操作系统、Python 版本、深度学习框架及版本、主要依赖库及版本等信息。

例如：

- 操作系统：在此填写
- Python：在此填写版本
- 深度学习框架：在此填写 MindSpore 或 PyTorch 及版本
- 其他依赖：在此填写
- 硬件：在此填写 CPU、GPU 或其他加速设备信息

= 4. 实验过程、结果及分析

== 4.1 理论基础与总体思路

在此介绍实验涉及的必要理论，并说明整体实现思路、关键公式或算法流程。

== 4.2 数据准备

在此说明数据集来源、数据规模、数据预处理方法、训练集/验证集/测试集划分方式，以及必要的数据可视化或统计结果。

== 4.3 模型与关键实现

在此说明模型结构、关键超参数、损失函数、优化器和训练设置。代码截图应清晰展示关键实现，并在正文中解释其作用。

#block(
  width: 100%,
  inset: 1em,
  fill: luma(245),
  stroke: 0.5pt + gray,
)[
  *代码截图占位：* 在此插入关键代码截图，并在下方说明截图中的实现。
]

== 4.4 训练过程与运行结果

在此记录训练过程、关键日志、评价指标和运行结果。运行结果截图应包含必要的上下文信息，并在正文中说明截图所反映的结果。

#block(
  width: 100%,
  inset: 1em,
  fill: luma(245),
  stroke: 0.5pt + gray,
)[
  *运行结果截图占位：* 在此插入程序输出、生成样例、预测结果或其他必要截图。
]

== 4.5 结果分析

在此分析实验结果，讨论损失和评价指标的变化、不同参数或方案的影响，并说明异常现象及其可能原因。

= 5. 实验总体结论

在此总结实验完成情况、主要结果、对理论知识的理解和实验中获得的结论。不要只重复实验过程，应明确回答实验目标是否达到。

= 6. 完整实验代码

在此粘贴可复现实验结果的完整代码。代码较长时，可按文件或功能分小节列出；代码截图不能替代完整代码。

#raw(block: true, lang: "python", "在此粘贴完整实验代码。")

= 7. 参考文献

按统一格式列出实验中实际参考的教材、论文、官方文档、数据集说明和其他资料。

+ 在此填写参考文献。
+ 在此填写参考文献。
