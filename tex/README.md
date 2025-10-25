# Create your paper

This folder contains the latex files for the Transformer Tricks papers. The flow is as follows:
1) Write first draft and drawings in Google docs.
2) Create file `foo.tex` and copy text from the Google doc.
    - Copy each drawing into a separate google drawing file and adjust the bounding box and "download" as PDF. This PDF is then used by latex.
    - For references, see the comments in file `references.bib`
3) Type `./run foo.tex` to create PDF.
4) Use spell checker as follows: `cd ..; util/spell_check`
5) Note: I converted some figures from PDF to SVG (so that I can use them in markdown) as follows `pdftocairo -svg foo.pdf foo.svg`  TODO: maybe only use SVG drawings even for tex.
6) Submit to arXiv:
    - To submit `foo.tex`, type: `./submit foo.tex`
    - To double-check if everything works, run `pdflatex foo` two times (or sometimes three times) as follows:
      `cd foo_submit` and `pdflatex foo && pdflatex foo`
   - Then upload the generated `*.tar.gz` file to arXiv.
   - Notes for filling out the abstract field in the online form:
     - Make sure to remove citations or replace them by `arXiv:YYMM.NNNNN`
     - You can add hyperlinks to the abstract as follows: `See https://github.com/blabla for code`
     - You can force a new paragraph in the abstract by typing a carriage return followed by one white space in the new line (i.e. indent the new line after the carriage return)
   - Keep in mind: papers that are short (e.g. 6 pages or less) are automatically put on `hold` and need to be reviewed by a moderator, which can take several weeks.

# Promote your paper
- Post on social media: LinkedIn, twitter
- Post on reddit and discord
- Generate a podcast and YouTube video:
  - We use Notebook LM to generate audio podcasts. We then manually create videos with this audio, see [here](https://www.youtube.com/@OpenMachine)
  - Try generating videos and podcasts with the [arXiv paper reader](https://github.com/imelnyk/ArxivPapers), see videos [here](https://www.youtube.com/@ArxivPapers)
 
# Submit to conference
- We don't have any experience with this
- It requires adding an introduction section and an extensive experiment section
