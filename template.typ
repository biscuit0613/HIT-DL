// 可复用的中文实验报告模板

#let three-line(columns, header, ..rows) = {
  table(
    columns: columns,
    align: center,
    stroke: none,
    inset: (x: 10pt, y: 6pt),
    table.hline(stroke: 1.2pt),
    table.header(..header),
    table.hline(stroke: 0.6pt),
    ..rows.pos().flatten(),
    table.hline(stroke: 1.2pt),
  )
}

#let report(
  experiment-number: "1",
  experiment-title: "实验题目",
  student-id: "待填写",
  student-name: "待填写",
  course-title: "《深度学习》实验报告",
  body,
) = {
  set document(
    title: course-title,
    author: student-name,
  )
  set page(
    paper: "a4",
    margin: (top: 2.25cm, bottom: 1.75cm, left: 2.6cm, right: 2.6cm),
    numbering: "1",
  )
  set text(
    font: ("Times New Roman", "Noto Serif CJK SC"),
    size: 10.5pt,
    lang: "zh",
  )
  set par(
    justify: true,
    leading: 0.65em,
    first-line-indent: (amount: 2em, all: true),
  )
  set heading(numbering: "1.1.1.")
  set block(spacing: 0.8em)

  show heading.where(level: 1): it => {
    block(above: 1.2em, below: 1.2em)[
      #text(size: 14pt, weight: "regular")[#it]
    ]
  }
  show heading.where(level: 2): it => {
    block(above: 1.2em, below: 1.2em)[
      #text(size: 12pt, weight: "regular")[#it]
    ]
  }
  show heading.where(level: 3): it => {
    block(above: 1.2em, below: 1.2em)[
      #text(size: 12pt, weight: "regular")[#it]
    ]
  }
  show raw.where(block: true): it => {
    block(fill: luma(245), inset: 8pt, radius: 2pt, width: 100%)[#it]
  }
  show raw: set text(font: "JetBrains Mono")
  show link: set text(fill: rgb("#245a9c"))

  align(center)[
    #text(size: 26pt, weight: "regular", font: "DengXian")[#course-title]
    #v(0cm)
    #text(size: 26pt, font: "DengXian")[（第 #experiment-number 次）]
    #v(5.2cm)
    #{
      show table.cell: set text(size: 16pt, font: "DengXian")
      table(
        columns: (3.2cm, 9cm),
        align: left,
        inset: 8pt,
        stroke: none,
        [实验题目：], [#experiment-title],
        [学号：], [#student-id],
        [姓名：], [#student-name],
      )
    }
  ]

  pagebreak()

  align(center)[
    #text(size: 16pt, weight: "bold", font: "DengXian")[实验报告内容]
  ]

  body
}
