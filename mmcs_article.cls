%--------------------------------------------------------------
% Class: mmcs-article
% author: E.V. Shiryaeva, sfedu. mmcs
% version: 1.11.05.2020
% Purpose: this LaTeX class provides an implementation
% of an article that standardizes the document layout
% intended for graduation papers.
%--------------------------------------------------------------
\NeedsTeXFormat{LaTeX2e}

\ProvidesClass{mmcs-article}[2020/05/10 v1.0.0 Traap LaTeX Class Article]

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}

\ProcessOptions

\LoadClass{article}

\RequirePackage{extsizes} %% обязательно перед geometry

\RequirePackage{misccorr}

%--------------------------------------------------------------
% Page layout
\RequirePackage[%reset,
                a4paper,
                top=2cm, %
                bottom=2cm, %
                left=3cm, %
                right=1.5cm,
                includehead=false,
                includefoot
                ]{geometry}

% установка межстрочного интервала
\linespread{1.3}
%\renewcommand{\baselinestretch}{1.30}

% установка абзацного отступа
\setlength\parindent{12.5mm}

\RequirePackage{graphicx} % вставка графики
\RequirePackage{float} % для плавающих объектов

%\usepackage[labelsep=period, position=top, textfont={normalsize}]{caption}
\usepackage[labelsep=period, position=top]{caption}

\usepackage[square,numbers,sort&compress]{natbib}
\renewcommand{\bibnumfmt}[1]{#1.\hfill} % нумерация источников в самом списке — через точку
\setlength{\bibsep}{0pt} \setlength{\bibindent}{2.5cm}

\RequirePackage{indentfirst}

\RequirePackage{setspace} % управление интервалами в тексте

\RequirePackage{cmap} %% для pdf
% ------------------------------------------------------------------------------
% \section etc.

\setcounter{tocdepth}{2}

\renewcommand\section{\@startsection {section}{1}{\z@}%
         {-3.5ex \@plus -1ex \@minus -.2ex}%
         {2.3ex \@plus.2ex}%
         {\leftskip12.5mm \normalfont\large\bfseries}}
				
\renewcommand\subsection{\@startsection {subsection}{1}{\z@}%
         {-3.5ex \@plus -1ex \@minus -.2ex}%
         {2.3ex \@plus.2ex}%
         {\leftskip12.5mm \normalfont\large\bfseries}}
				
\renewcommand\subsubsection{\@startsection {subsubsection}{1}{\z@}%
         {-3.5ex \@plus -1ex \@minus -.2ex}%
         {2.3ex \@plus.2ex}%
         {\leftskip12.5mm \normalfont\normalsize\bfseries}}

%\def\@biblabel#1{#1.}

% ------------------------------------------------------------------------------
% Toc etc.

\RequirePackage{tocloft}
\renewcommand\cftsecdotsep{\cftdot} % {\cftnodots}
\renewcommand{\cftsecpagefont}{\normalfont} % {\bfseries}
\renewcommand{\cftsecfont}{\normalfont} % {\bfseries}
\renewcommand{\cftsecleader}{\cftdotfill{\cftsecdotsep}}
%\newcommand{\cftsecleader}{\bfseries\cftdotfill{\cftsecdotsep}}

\renewcommand{\cfttoctitlefont}{\hfil\Large\bfseries}
\renewcommand{\cftaftertoctitle}{\hfil}

\renewcommand\cftsubsecdotsep{\cftdot}
\renewcommand{\cftsubsecfont}{\normalfont}

\setlength{\cftbeforesecskip}{\z@ \@plus.2\p@}
\setlength{\cftbeforesubsecskip}{\z@ \@plus.2\p@}

% ------------------------------------------------------------------------------
% listings
% листинги для разных языков программирования будут одинаковые,
% см. basicstyle

\RequirePackage{listings}

\lstset{basicstyle=\ttfamily\fontsize{12pt}{12pt}\selectfont,
showstringspaces=false, % не показывать пробелы в строках
extendedchars=\true, commentstyle=\ttfamily, keepspaces=true }

% ------------------------------------------------------------------------------
% переопределение списка itemize
\renewcommand{\@listI}{%
\topsep=0pt }

\let\old@itemize=\itemize
\def\itemize{\old@itemize
 \setlength{\itemsep}{0pt}
 \setlength{\parskip}{0.0pt}
 \setlength{\leftskip}{0.8cm}
}
