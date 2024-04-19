This folder contains all files to generate the PDF of the Transformer Tricks papers. The flow is as follows:
1) Write first draft and drawings in Google docs.
2) Create file `foo.tex` and copy text from the Google doc.
    - For drawings, copy the drawings into a google drawings file and adjust the bounding box, and then "download" as PDF. This PDF is then used by latex.
    - For references, see the comments in file `references.bib`.
3) Upload to arXiv:
    - To submit `foo.tex`, type: `./submit foo`.
    - To double-check if everything works, run `pdflatex foo` two times (or sometimes three times) as follows:
      `cd foo_submit` and `pdflatex foo && pdflatex foo`.
   - Then upload the generated `*.tar.gz` file to arXiv.
   - Notes for filling out the abstract field in the online form:
     - Make sure to remove citations or replace them by `arXiv:YYMM.NNNNN`.
     - You can add hyperlinks to the abstract as follows: `See https://github.com/blabla for code`.
   - Keep in mind: papers that are short (e.g. 6 pages or less) are automatically put on `hold` and need to be reviewed by a moderator, which can take several weeks.
