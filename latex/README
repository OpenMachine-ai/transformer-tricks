This folder has all files to generate the PDF of the Transformer Tricks papers. The flow is as follows:
1) Write first draft and drawings in Google docs
2) Create a foo.tex file and copy & paste stuff from Google docs. For drawings,
   you need to copy the drawings into a google drawings file and adjust
   the bounding box, and then "download" as PDF. That PDF is then used by latex.
   For references, see comments in references.bib
3) Upload to arXiv:
   To submit foo.tex, type:
   ./submit foo
   # Note: to double-check if everything works, run 'pdflatex foo' twice as follows:
   #  cd foo_submit
   #  pdflatex foo && pdflatex foo
   - Then upload this tar.gz file on arXiv
   - When you enter the "abstract" in the online form, make sure to remove
     citations or replace them by arXiv:YYMM.NNNNN
   - You can add hyperlinks to the abstract as follows:
       "See https://github.com/blabla for code."
   - Keep in mind: papers that are short (e.g. 6 pages) are automatically put on hold and then
     a moderator has to review the paper, which can take several weeks