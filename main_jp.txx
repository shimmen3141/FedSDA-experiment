% !TEX recipe = latexmk (LuaLaTeX) 🌟日本語版用
% これは main_jp.tex です。
% Springer Computer Science proceedings 用の LLNCS マクロパッケージを
% 使用した日本語版のサンプルファイルです。

\documentclass[runningheads]{llncs}

% ==========================================
% 日本語化のための追加設定 (LuaLaTeX用)
% ==========================================
\usepackage{luatexja} % 日本語を使用するための基本パッケージ
\usepackage[haranoaji]{luatexja-preset} % 標準的な日本語フォント（原ノ味フォント）を指定
\renewcommand{\figurename}{Fig.} % 図のキャプションを「図」ではなく「Fig.」にする場合（学会規定に合わせて変更してください）
\renewcommand{\tablename}{Table} % 表のキャプション
% ==========================================

\usepackage[T1]{fontenc}
% ハイパーリンクを使用する場合は以下のコメントアウトを外してください
%\usepackage{color}
%\renewcommand\UrlFont{\color{blue}\rmfamily}
%\urlstyle{rm}

\usepackage[fleqn]{amsmath}
\usepackage[psamsfonts]{amssymb}
%\usepackage[deluxe]{otf}
\usepackage{algorithmic}
\usepackage{algorithm}
\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{makecell}
\usepackage{url}

\usepackage{graphicx,xcolor}
\usepackage{tikz}
\usetikzlibrary{shapes.geometric, arrows, positioning, calc, shapes.misc, backgrounds}
\usepackage{subcaption}

% argmin, argmax の定義
\DeclareMathOperator*{\argmin}{argmin}
\DeclareMathOperator*{\argmax}{argmax}

% 疑似コードの設定
\renewcommand{\algorithmicrequire}{\textbf{Input:}}
\renewcommand{\algorithmicensure}{\textbf{Output:}}
\newcommand{\INTERNALSTATE}[1]{\item[\textbf{Internal State:}] #1}
\newcommand{\PARAMETERS}[1]{\item[\textbf{Parameters:}] #1}


\begin{document}

\title{統計的検出によるローカルなコンセプトドリフトに対応可能な連合学習法}

%\titlerunning{省略された論文タイトル}
% タイトルが長すぎてヘッダーに収まらない場合は、ここで省略形を設定します

\author{第一 著者\inst{1}\orcidID{0000-1111-2222-3333} \and
第二 著者\inst{2,3}\orcidID{1111-2222-3333-4444} \and
第三 著者\inst{3}\orcidID{2222--3333-4444-5555}}

\authorrunning{第一著者ほか} % 著者名が長い場合は省略形を使用します

\institute{〇〇大学 大学院情報科学研究院, 北海道 札幌市 \\
\email{author1@example.com} \and
Springer Heidelberg, Tiergartenstr. 17, 69121 Heidelberg, Germany\\
\email{lncs@springer.com}\\
\url{http://www.springer.com/gp/computer-science/lncs} \and
ABC Institute, Heidelberg, Germany\\
\email{\{abc,lncs\}@uni-heidelberg.de}}

\maketitle

\begin{abstract}
連合学習\cite{mcmahan2017}は，複数クライアントがローカルデータを用いてモデルを学習し，そのモデル情報を中央サーバに集約することで，効率的かつプライバシーを保護した学習を行う手法である．連合学習においてより高い性能を達成するためには，各クライアントが自身のローカルデータに適したモデルを使用し，コンセプトドリフトが発生した際にはより適切なモデルへと切り替える必要がある．本稿では，連合学習の性能を向上させるため，ADWIN~\cite{bifet2007}と呼ばれる統計的なコンセプトドリフト検出手法を組み込んだ手法を提案する．実験の結果，提案手法はコンセプトドリフトへの適応速度およびモデル精度の観点において，既存手法\cite{jothimurugesan2023}を上回ることを示す．

\keywords{連合学習 \and コンセプトドリフト \and ADWIN.}
\end{abstract}

\section{はじめに}

近年, スマートフォンやIoTデバイスの普及に伴い, 膨大かつ分散したデータが端末側に蓄積されるようになった.これらのデータは多くの個人情報を含む可能性があり, 中央サーバへ生データを送信せずに学習を行うことが求められている.連合学習（Federated Learning）~\cite{mcmahan2017} はこれに応える枠組みとして注目されており, 複数のクライアントがローカルで勾配等のモデル更新を計算し, その情報のみを中央サーバへ送ることで全体モデルを学習する手法である. 連合学習はプライバシー保護や通信コスト削減といった利点を持つ一方で, 時間とともに生じるデータ分布の変化（コンセプトドリフト）への対応ができず, 学習の収束性・性能が損なわれやすいという課題がある.

こうした課題に対して, クライアント側のコンセプトドリフトに焦点を当てた連合学習手法が近年提案されている. なかでもFedDrift~\cite{jothimurugesan2023}は, 各クライアントで発生するローカルなコンセプトドリフトを検出し, コンセプトをクラスター化（同種のコンセプトを共有するクライアント群の同定）して連合学習を進めるアプローチである. コンセプトドリフトを考慮することによって, 実社会におけるデータに対してもモデルの性能維持や迅速な適応を目指している.しかしながら, ドリフトの検出やクラスタリングに関してはヒューリスティックに頼る部分が残っており, 統計的な妥当性や理論的保証が十分ではない. 

本稿では, FedDriftの枠組みをベースに, ストリーム上での平均変化を自動適応ウィンドウによって検出する統計アルゴリズムであるADWIN~\cite{bifet2007}を導入する. ADWINは, 入力されるストリームデータに対して可変長なウィンドウを持ち, ウィンドウ内の全ての可能な分割点について, 2つのサブウィンドウ間に統計的に有意な差があるかを逐次的に確認する手法である. サブウィンドウ間の有意差を判定する閾値は, データに応じて統計的に導出される. 

提案手法では, ADWINの導入によってコンセプトドリフトの検出に対して統計的根拠に基づく改良を与えることで, より迅速にローカルなコンセプトドリフトへ対応することを目的とする. 具体的には以下の点を貢献として挙げる.

\begin{itemize}
    \item ADWINを用いたローカルドリフト検出機構の導入により, ドリフト発生時の検出精度を改善し, 検出アルゴリズムに対して統計的根拠を与える.
    \item 提案手法が既存のFedDriftと比較して, ドリフトへの適応速度や全体モデルの性能面で優れることを実験的に示す.
\end{itemize}

\section{関連研究}

\subsection{連合学習とデータ異質性}

連合学習は，複数のクライアントがデータを中央サーバに集約せず，ローカルで計算したモデル更新を共有して協調的にモデルを学習する枠組みである．代表的なアルゴリズムであるFederated Averaging（FedAvg）~\cite{mcmahan2017}では，各クライアントが共通モデルをローカルデータで複数回更新し，サーバが更新後のモデルをデータ数に基づいて加重平均することで，通信回数を抑えながら学習を行う．

実際の連合学習では，クライアントごとにデータ量，ラベル分布および特徴分布が異なる非独立同分布（non-IID）データが一般的である~\cite{kairouz2021}．このような統計的異質性は，FedAvgの収束を遅延または不安定化させる場合がある．FedProx~\cite{li2020fedprox}は，ローカル目的関数にグローバルモデルとの距離に基づく近接項を加えることで，統計的異質性とクライアントごとに実行可能なローカル更新量の違いに対処する．ただし，FedAvgとFedProxは単一のグローバルモデルを学習する枠組みであり，時間とともに変化するクライアント内のデータ分布を明示的に扱わない．

\subsection{クラスタリングおよび複数モデルを用いる連合学習}

クライアント間の異質性が大きい場合，類似するクライアントをクラスタリングし，クラスタごとに異なるモデルを学習する方法が用いられる．

その手法の一つであるIterative Federated Clustering Algorithm（IFCA）~\cite{ghosh2020ifca}は，複数の潜在的なクライアントクラスタを仮定し，クライアントのクラスタ割当てとクラスタモデルの更新を交互に行う．静的な異質性に対応する代表的なclustered federated learning手法であるが，クラスタ数を事前に与える必要があり，時間とともに未知のコンセプトが出現する状況は直接扱わない．

これに対し，FedDrift~\cite{jothimurugesan2023}は，各クライアントの分布が時間とともに変化するdistributed concept driftを対象とする．複数のグローバルモデルを保持し，ローカルなドリフト検知に基づいて新規モデルを生成するとともに，階層的クラスタリングによって類似するクライアントクラスタとモデルを統合する．

同じくdistributed concept driftを対象とするFedCCFA~\cite{chen2024fedccfa}は，ローカル分類器のクラスごとのパラメータをクラスタリングし，生成したfeature anchorによってクライアント間の特徴表現を整列させる．FedCCFAが分類器と特徴表現の共有を主眼とするのに対し，本研究ではドリフトを統計的に検知し，複数モデルの生成とデータ割当てを制御することに着目する．

\subsection{コンセプトドリフトとドリフト適応}

コンセプトドリフトは，時間経過に伴ってデータ生成分布が変化する現象であり，時刻$t$における同時分布$P_t(X,Y)$の変化として定式化される~\cite{gama2014survey,lu2019review}．条件付き分布$P(Y\mid X)$が変化する場合はreal drift，$P(Y\mid X)$が不変のまま入力分布$P(X)$が変化する場合はvirtual driftと呼ばれ，機械学習ではcovariate shiftとして扱われる．

コンセプトドリフトへの対応は，変化を判定するdrift detection，変化の種類や原因を分析するdrift understanding，およびモデルを更新・置換するdrift adaptationに大別される~\cite{lu2019review}．本研究では，ADWINがdrift detectionを担い，FedDriftに基づくモデルの生成，選択およびデータ割当てがdrift adaptationを担う．このうちdrift adaptationに関する手法であるDriftSurf~\cite{tahmasbi2021driftsurf}は，安定状態と反応状態を設け，性能低下の検知後に既存モデルと新たに学習したモデルを比較することで，一時的な性能低下や誤検知による不要なモデル切替えを抑制する．

\subsection{ドリフト検知手法}

一方，代表的なdrift detection手法であるDrift Detection Method（DDM）~\cite{gama2004}は，各予測の正誤をBernoulli変数として扱い，オンライン分類器の誤り率と標準偏差を監視する．それらの過去の最良値を基準として，警告レベルを超えた場合に警告状態へ移行し，ドリフトレベルを超えた場合に変化を判定する．

同じくdrift detection手法であるADWIN（ADaptive WINdowing）~\cite{bifet2007}は，可変長ウィンドウを用いて統計的に変化を検知する．ウィンドウを古い部分と新しい部分に分割し，両者の平均値の差が統計的閾値を超えた場合に古い部分を削除することで，観測数や観測値の分散に応じて検知基準を調整する．本研究では，FedDriftの固定閾値に基づく検知機構をADWINによる統計的変化検知に置き換え，クライアントごとの損失系列からドリフトを検知する．これにより，信頼度パラメータに基づく検知と複数モデルによる適応を組み合わせる．

\section{問題設定}

本稿では，$C$ 個のクライアントと1つの中央サーバからなる連合学習システムを考える．クライアントの集合を $\mathcal{C} = \{1, \dots, C\}$ とおく．各時刻（タイムステップ） $t=1,2,\dots$ において，各クライアント $c \in \mathcal{C}$ は特徴データ $x_c^{(t)}$ を受け取り，その予測ラベル $\hat{y}_c^{(t)}$ を出力する．その後，真のラベル $y_c^{(t)}$ が明らかになり，クライアント $c$ は損失 $\ell(\hat{y}_c^{(t)},y_c^{(t)})$ を受ける．システム全体の目的は，各クライアントに到着するデータに対する損失の合計を最小化することである．

データ $(x_c^{(t)},y_c^{(t)})$ はある分布 $\mathcal{D}_c^{(t)}$ に従って生成されると仮定する．この分布は時間経過とともに変化する可能性があり，すなわちコンセプトドリフト（$\mathcal{D}_c^{(t)} \neq \mathcal{D}_c^{(t+1)}$）が発生し得る．また，$c \neq c'$ のとき，分布 $\mathcal{D}_c^{(t)}$ は $\mathcal{D}_{c'}^{(t)}$ と異なる可能性がある．

\subsection{問題の定式化}
本研究では以下の仮定を置く．
\begin{enumerate}
    \item \textbf{損失の有界性:} モデル $h$ の予測に対する損失関数 $\ell(h(x), y)$ は，区間 $[0, 1]$ に収まるものとする\footnote{Bernsteinの不等式による理論保証を可能にする条件.}（必要であれば正規化を行う）．
    \item \textbf{データの独立性:} 各ステップ $t$，各クライアント $c$ において，データ $(x_c^{(t)},y_c^{(t)})$ は分布 $\mathcal{D}_c^{(t)}$ から独立にサンプリングされる．
\end{enumerate}

サーバは通信ラウンド $r$ において $\left|\mathcal{H}^{[r]}\right|$ 個のグローバルモデルの集合 $\mathcal{H}^{[r]} = \{h_1, \dots, h_{\left|\mathcal{H}^{[r]}\right|}\}$ を保持・管理する．
各クライアント $c$ は通信ラウンド $r$ の最初にサーバから $\mathcal{H}^{[r]}$ をダウンロードし，自身のローカルモデル集合 $\mathcal{H}_c^{(t)}$ を初期化する．すなわち，ラウンド開始時点では $\mathcal{H}_c^{(t)} = \mathcal{H}^{[r]}$ である．クライアントは $r$ 中の各時刻 $t$ において，$\mathcal{H}_c^{(t)}$ の中から自身の現在のデータ分布に最適なモデルを選択したり，モデルの更新・新規モデルの追加を行ったりする．

ここで，時刻 $t$ における全クライアントのモデル選択をベクトル $m^{(t)} = (m_1^{(t)}, \dots, m_C^{(t)})$ とし，各要素 $m_c^{(t)}$ はクライアント $c$ が $\mathcal{H}_c^{(t)}$ から選択したモデルIDを表すことにする．したがって，$h_{m_c^{(t)}}$ は時刻 $t$ にクライアント $c$ が選択するモデルを意味し， $h_{m_c^{(t)}} \in \mathcal{H}_c^{(t)}$ である．

システム全体の目的は，モデルの集合列 $(\mathcal{H}_c^{(t)})_{c, t}$ と各クライアントのモデル選択列 $(m^{(t)})_{t}$ を最適化して，全期間 $t=1, \dots, T$ にわたる全クライアントの期待損失の総和を最小化することである．

\begin{equation}
\min_{(\mathcal{H}_c^{(t)})_{c, t}, (m^{(t)})_{t}} \sum_{t=1}^T \sum_{c \in \mathcal{C}} \mathbb{E}_{(x, y) \sim \mathcal{D}_c^{(t)}}[\ell(h_{m_c^{(t)}}(x), y)]
\end{equation}

分布 $\mathcal{D}_c^{(t)}$ やその変化点は未知であるため，クライアント $c$ は順次観測されるデータストリームから自律的に $m^{(t)}_c$ を決定し，必要であれば新規モデルをクライアント側で作成することになる．作成されたモデルや更新されたパラメータは，通信ラウンドごとにサーバへ送信される．サーバではモデルの統合（マージ）等の処理が行われ，次ラウンドのための新たなグローバルモデル集合 $\mathcal{H}^{[r+1]}$ が構築される．

\section{提案手法}
まず，提案手法の概略を説明する．クライアントはサーバからモデル集合を受け取った後，各時刻 $t$ において，到着するデータストリームから時系列順に1個のデータ $(x_c^{(t)},y_c^{(t)})$ を取得する．このデータを用いて現在選択中のモデルで損失計算およびドリフト検出を行い，ローカル更新を実行する．このデータの取得と一連の処理をローカルステップと呼ぶことにする．各クライアントで $K$ 回のローカルステップを経た後，モデルの更新データ等がサーバに送信され，サーバでの加重平均やモデルの評価のための通信・クラスタリングが行われる．この一連の処理を通信ラウンドと呼び，システム全体は通信ラウンド $r = 1, 2, \dots$ を一つの単位として進行する．ラウンド $r$ における処理・通信の流れを図\ref{fig:sequence_diagram}に示す．次節では，提案手法の根幹部分であるクライアント側のアルゴリズムについて詳述する．

\begin{figure}[htbp]
    \centering
    \resizebox{\textwidth}{!}{
        \begin{tikzpicture}[
            >=stealth, % 矢印の形状
            thick,
            every node/.style={align=center},
            box/.style={draw, fill=white, minimum width=2.5cm, minimum height=0.6cm},
            bluebox/.style={draw, fill=cyan!10, minimum height=0.6cm, inner sep=5pt},
            greenbox/.style={draw, fill=green!10, minimum width=5cm, minimum height=0.8cm, inner sep=6pt},
            yellowbox/.style={draw, fill=yellow!15, minimum width=4cm, minimum height=0.8cm, inner sep=6pt}
        ]
        
        % --- X座標の定義 ---
        \def\xc{0}   % クライアントのX座標
        \def\xs{9}   % サーバのX座標
        \def\loopleft{-4.0}  % ループ枠の左端
        \def\loopright{4.0}  % ループ枠の右端
        
        % --- Y座標の定義（直前の座標からの相対計算レイアウト） ---
        \def\step{1.5}       % 基本のノード間隔
        \def\largestep{2.0}  % 大きめのノード間隔（枠が大きい処理用）
        \def\smallstep{1.0}  % ループ枠周りの少し狭い間隔

        \def\yhead{0} % アクター
        \pgfmathsetmacro{\ybcst}{\yhead - \smallstep} % ブロードキャスト
        \pgfmathsetmacro{\yinit}{\ybcst - \smallstep} % クライアント初期化
        \pgfmathsetmacro{\looptop}{\yinit - \smallstep} % ループ枠の上端
        \pgfmathsetmacro{\ydata}{\looptop - \step} % データ到着
        \pgfmathsetmacro{\yproc}{\ydata - 1.8} % ローカル処理
        \pgfmathsetmacro{\loopbot}{\yproc - \step} % ループ枠の下端
        \pgfmathsetmacro{\yextr}{\loopbot - \step} % 抽出処理
        \pgfmathsetmacro{\yupld}{\yextr - \step} % アップロード
        \pgfmathsetmacro{\yfedavg}{\yupld - \step} % 加重平均 (枠が大きいため間隔を2.0に)
        \pgfmathsetmacro{\yevalreq}{\yfedavg - \step} % 評価依頼
        \pgfmathsetmacro{\yevalproc}{\yevalreq - \smallstep} % 評価計算
        \pgfmathsetmacro{\yevalres}{\yevalproc - \smallstep} % 評価返信
        \pgfmathsetmacro{\ymerge}{\yevalres - \step} % マージ処理
        \pgfmathsetmacro{\ybottom}{\ymerge - \step} % ライフラインの下端
        
        % ==========================================
        % 0. ライフライン（Zオーダーを最背面にするため最初に描画）
        % ==========================================
        \draw[gray, thick] (\xc, \yhead) -- (\xc, \ybottom);
        \draw[gray, thick] (\xs, \yhead) -- (\xs, \ybottom);
        
        % 続くことを示す破線
        \draw[gray, thick, dashed] (\xc, \ybottom) -- (\xc, \ybottom - 1.0);
        \draw[gray, thick, dashed] (\xs, \ybottom) -- (\xs, \ybottom - 1.0);

        % --- ヘッダー（アクター） ---
        \node[box] at (\xc, \yhead) {クライアント $c$};
        \node[box] at (\xs, \yhead) {サーバ};
        
        % ==========================================
        % 1. ブロードキャストと初期化
        % ==========================================
        \draw[->, red, very thick] (\xs, \ybcst) -- node[above, text=black] {初期モデル集合 $\mathcal{H}^{[r]}$} (\xc, \ybcst);
        \node[greenbox] at (\xc, \yinit) {ワーキング集合の初期化\\$\mathcal{H}_{c, \text{work}}^{(t_0)} \leftarrow \mathcal{H}^{[r]}$};

        % ==========================================
        % 2. UML準拠のK回ループ枠
        % ==========================================
        % メインの矩形枠
        \draw[thick, draw=blue!40!black] (\loopleft, \looptop) rectangle (\loopright, \loopbot);
        
        % 動的サイズの五角形タブ
        \node[anchor=north west, font=\sffamily\small, inner xsep=6pt, inner ysep=4pt] 
            (lt) at (\loopleft, \looptop) {\textbf{loop} [ $K$ times : 各時刻 $t$ ]};
        \draw[thick, draw=blue!40!black, fill=white]
            (lt.north west) -- 
            (lt.south west) -- 
            (lt.south east) -- 
            ([xshift=0.2cm, yshift=0.2cm]lt.south east) -- % 右下を斜めにカット
            ([xshift=0.2cm]lt.north east) -- 
            cycle;
        % タブの上にテキストを再描画（文字が線に隠れないように）
        \node[anchor=north west, font=\sffamily\small, inner xsep=6pt, inner ysep=4pt] 
            at (\loopleft, \looptop) {\textbf{loop} [ $K$ times : 各時刻 $t$ ]};
        
        % --- ループ内部の処理 ---
        \node[bluebox] (data) at (\xc-2.5, \ydata) {データ $(x_c^{(t)},y_c^{(t)})$};
        \draw[->, red, very thick] (data.east) -- (\xc, \ydata); % 矢印をライフラインに当てる
        
        \node[greenbox] at (\xc, \yproc) {損失計算・ドリフト検知\\$\downarrow$\\$\mathcal{H}_{c, \text{work}}^{(t)}$ から最適モデルを選択・新規作成し\\ローカル更新};

        % ==========================================
        % 3. クライアント：抽出とアップロード
        % ==========================================
        \node[greenbox] at (\xc, \yextr) {最終状態 $\mathcal{H}_{c, \text{work}}^{(rK)}$ から抽出\\$\downarrow$\\更新済モデル $\mathcal{H}_{c,\text{upd}}^{[r]}$, 新規モデル $\mathcal{H}_{c,\text{new}}^{[r]}$};

        \draw[->, red, very thick] (\xc, \yupld) -- node[above, text=black] {$\mathcal{H}_{c,\text{upd}}^{[r]}, \mathcal{H}_{c,\text{new}}^{[r]}$} (\xs, \yupld);
        
        % ==========================================
        % 4. サーバ：FedAvg と評価依頼
        % ==========================================
        \node[yellowbox, text width=4.5cm] at (\xs, \yfedavg) {
            \textbf{【FedAvgによる加重平均】}\\[0.3em]
            $\mathcal{H}_{\text{upd}}^{[r]} \leftarrow \text{FedAvg}(\mathcal{H}_{c,\text{upd}}^{[r]})$\\[0.5em]
            $\mathcal{H}_{\text{new}}^{[r]} \leftarrow \displaystyle\bigcup_{c \in \mathcal{C}} \mathcal{H}_{c,\text{new}}^{[r]}$
        };

        \draw[->, red, very thick] (\xs, \yevalreq) -- node[above, text=black] {評価依頼 $\mathcal{H}_{\text{upd}}^{[r]} \cup \mathcal{H}_{\text{new}}^{[r]}$} (\xc, \yevalreq);
        
        % ==========================================
        % 5. 評価値の計算とマージ処理
        % ==========================================
        \node[greenbox] at (\xc, \yevalproc) {受信した $\mathcal{H}_{\text{upd}}^{[r]} \cup \mathcal{H}_{\text{new}}^{[r]}$ を\\ローカルデータで評価};

        \draw[->, red, very thick] (\xc, \yevalres) -- node[above, text=black] {評価結果（ロス等）} (\xs, \yevalres);
        
        \node[yellowbox, text width=4.5cm] at (\xs, \ymerge) {
            \textbf{【クラスタリング・統合】}\\[0.3em]
            評価値を集計し距離行列を構成\\
            $\downarrow$\\
            階層的クラスタリング・マージ\\
            $\rightarrow \mathcal{H}^{[r+1]}$ を構築
        };

        \end{tikzpicture}
    }
    \caption{ラウンド $r$ における提案システムのシーケンス図}
    \label{fig:sequence_diagram}
\end{figure}

\subsection{クライアントアルゴリズム}

本節では，提案システムにおけるクライアント側のアルゴリズムについて詳述する．クライアントは，ADWINを用いた統計的なドリフト検出によるモデル選択を行うことで，計算コストを抑えつつコンセプトドリフトに柔軟に適応する．クライアントアルゴリズムは，全体としてADWINによるドリフト検出とデータストアへのデータ保存によって構成され，ADWINによって発生しうる不整合をFIFOバッファを用いて調整する．このクライアントアルゴリズムを Algorithm \ref{alg:client_main} として示す．なお，ドリフト検出後のモデル選択や新規モデルの作成については，後述するサブルーチン RESOLVEDRIFT に切り分けている．また，アルゴリズムを記述するうえで用いるADWIN・FIFOバッファのメソッドを表\ref{tab:data_structures}にまとめる．それぞれ，信頼度パラメータ $\delta_{\text{adwin}}$ とFIFOバッファ長 $N_{\text{FIFO}}$ をパラメータとして持つ．

\begin{table}[htbp]
\centering
\caption{Algorithm \ref{alg:client_main} で使用するデータ構造とメソッド}
\label{tab:data_structures}
\small
\begin{tabularx}{\columnwidth}{l c c @{\hspace{0.5em}} X}
\toprule
\textbf{メソッド} & \textbf{入力} & \textbf{出力} & \textbf{動作の説明} \\
\midrule
\multicolumn{4}{l}{\textbf{ADWIN検出器 $\mathcal{A}$}} \\
\midrule
$\mathcal{A}.insert(\ell)$ & 損失 $\ell$ & なし & 内部の動的ウィンドウ $W$ の末尾に新しい損失 $\ell$ を追加し，ウィンドウを拡張する。 \\
\addlinespace
$\mathcal{A}.detect\_drift()$ & なし & 真偽値 & $W$ 内の全ての分割点で $\delta_{\text{adwin}}$ を用いて統計的にドリフトを検証し，閾値を超える分割が存在したら \texttt{True} を返す。 \\
\addlinespace
$\mathcal{A}.get\_split()$ & なし & \makecell[t]{データID集合 \\ $\mathcal{I}_{\text{old}}, \mathcal{I}_{\text{new}}$} & ドリフトが検知された分割点に基づき，古い概念と新しい概念のデータID集合を分割して返す。 \\
\addlinespace
$\mathcal{A}.shrink()$ & なし & なし & $W$ から $\{ \ell_i \in W \mid i \in \mathcal{I}_{\text{old}} \}$ を削除し，ウィンドウを縮小する。 \\
\midrule
\multicolumn{4}{l}{\textbf{FIFOバッファ $\mathcal{F}$}} \\
\midrule
$\mathcal{F}.enqueue(x, y)$ & データ $(x, y)$ & なし & バッファの末尾にデータ $(x, y)$ を追加する。 \\
\addlinespace
$\mathcal{F}.dequeue()$ & なし & データ $(x_{\text{out}}, y_{\text{out}})$ & バッファの先頭から最も古いデータを取り出し，それを返す。 \\
\addlinespace
$\mathcal{F}.clear()$ & なし & なし & バッファ内のデータを全て削除し，空の状態にする。 \\
\bottomrule
\end{tabularx}
\end{table}

\begin{algorithm}[htbp]
\caption{$\textsc{ClientProcess}$ at client $c$ in round $r$}
\label{alg:client_main}
\begin{algorithmic}[1]
\small
\REQUIRE
サーバからのグローバルモデル集合 $\mathcal{H}^{[r]}$
\INTERNALSTATE{現在選択中のモデルID $m_c$, ADWIN検出器 $\mathcal{A}_c$, 各モデル用データストアの集合 $\mathcal{S}_c = \{\mathcal{S}_{c, m}\}_m$, FIFOバッファ $\mathcal{F}_c$}
\PARAMETERS{ローカルステップ数 $K$, ローカル更新間隔 $\tau$, ローカル更新回数 $L$, 初期学習エポック $E_{\text{init}}$, 信頼度パラメータ $\delta_{\text{adwin}}$, FIFOバッファ長 $N_{\text{FIFO}}$}

\vspace{1em}

\STATE $t_0 \gets (r-1)K, \quad \mathcal{H}_{c, \text{work}}^{(t_0)} \gets \mathcal{H}^{[r]}$
\STATE $h_{m_c} \gets \mathcal{H}_{c, \text{work}}^{(t_0)}[m_c]$

\STATE \COMMENT{--- ローカルステップ ---}
\FOR{各時刻 $t = t_0 + 1, \dots, rK$}
    \STATE 今ステップのワーキング集合を準備: $\mathcal{H}_{c, \text{work}}^{(t)} \gets \mathcal{H}_{c, \text{work}}^{(t-1)}$
    \STATE データストリームからデータ $(x_c^{(t)}, y_c^{(t)})$ を取得

    \STATE 損失 $\ell_c^{(t)} \gets \ell(h_{m_c}(x_c^{(t)}), y_c^{(t)})$ を計算
    \STATE $\mathcal{A}_c.insert(\ell_c^{(t)}), \quad \mathcal{F}_c.enqueue(x_c^{(t)}, y_c^{(t)})$

    \IF{$\mathcal{A}_c.detect\_drift()$}
        \STATE $\mathcal{I}_{\text{old}}, \mathcal{I}_{\text{new}} \gets \mathcal{A}_c.get\_split()$
        \STATE $\mathcal{F}_{c, \text{old}} \gets \{(x_c^{(s)}, y_c^{(s)}) \in \mathcal{F}_c \mid s \in \mathcal{I}_{\text{old}}\}, \quad \mathcal{S}_{c, m_c} \gets \mathcal{S}_{c, m_c} \cup \mathcal{F}_{c, \text{old}}$
        \STATE $\mathcal{F}_{c, \text{new}} \gets \{(x_c^{(s)}, y_c^{(s)}) \in \mathcal{F}_c \mid s \in \mathcal{I}_{\text{new}}\}$

        \STATE $m_c \gets \textsc{ResolveDrift}(\mathcal{H}_{c, \text{work}}^{(t)}, \mathcal{F}_{c, \text{new}})$

        \IF{$m_c$ が新規ID}
            \STATE $\mathcal{S}_{c, m_c} \gets \mathcal{F}_{c, \text{new}},\quad \mathcal{S}_c \gets \mathcal{S}_c \cup \{\mathcal{S}_{c, m_c}\}$

            \STATE 既存モデルのパラメータから $h_{m_c}$ を初期化
            \STATE $h_{m_c}$ に対して $\mathcal{F}_{c, \text{new}}$ を用いて $E_{\text{init}}$ エポックの初期学習を実行
            \STATE $\mathcal{H}_{c, \text{work}}^{(t)} \gets \mathcal{H}_{c, \text{work}}^{(t)} \cup \{h_{m_c}\}$
        \ELSE
            \STATE $\mathcal{S}_{c, m_c} \gets \mathcal{S}_{c, m_c} \cup \mathcal{F}_{c, \text{new}}$
        \ENDIF

        \STATE $\mathcal{A}_c.shrink(), \quad \mathcal{F}_c.clear()$

    \ELSIF{$|\mathcal{F}_c| > N_{\text{FIFO}}$}
        \STATE $(x_{\text{out}}, y_{\text{out}}) \gets \mathcal{F}_c.dequeue(), \quad \mathcal{S}_{c, m_c} \gets \mathcal{S}_{c, m_c} \cup \{(x_{\text{out}}, y_{\text{out}})\}$
    \ENDIF

    \IF{$t \bmod \tau = 0$}
        \FOR{$|\mathcal{S}_{c, m}| > 0$ を満たす各 $m$}
            \STATE $\mathcal{S}_{c, m}$ からミニバッチをサンプリングし，$h_m$ に対して $L \cdot \tau$ 回のローカル更新を実行
            \STATE $\mathcal{H}_{c, \text{work}}^{(t)}$ 内の $h_m$ のパラメータを，更新後の状態に上書きする
        \ENDFOR
    \ENDIF
\ENDFOR

\STATE \COMMENT{--- サーバ送信用の抽出 ---}
\STATE $\mathcal{I}^{[r]} \gets \{ m \mid h_{m} \in \mathcal{H}^{[r]} \}$ \COMMENT{ラウンド開始時の既存モデルID集合}
\STATE $\mathcal{H}_{c, \text{upd}}^{[r]} \gets \left\{ h_{m} \in \mathcal{H}_{c, \text{work}}^{(rK)} \mid m \in \mathcal{I}^{[r]} \land |\mathcal{S}_{c, m}| > 0 \right\}$ \COMMENT{更新済み既存モデル集合}
\STATE $\mathcal{H}_{c, \text{new}}^{[r]} \gets \left\{ h_{m} \in \mathcal{H}_{c, \text{work}}^{(rK)} \mid m \notin \mathcal{I}^{[r]} \right\}$ \COMMENT{新規モデル集合}

\RETURN $\mathcal{H}_{c, \text{upd}}^{[r]}, \mathcal{H}_{c, \text{new}}^{[r]}$ \COMMENT{サーバへまとめて送信}
\end{algorithmic}
\end{algorithm}

次に，Algorithm \ref{alg:client_main} の動作について説明する．
まず，ラウンド開始時にサーバから受け取ったグローバルモデル集合 $\mathcal{H}^{[r]}$ でローカルのワーキング集合を初期化し，現在選択中のモデルID $m_c$ に対応するパラメータを同期する（1〜2行目）．
次に，ローカルステップとして各時刻 $t$ にデータストリームから順次データ $(x_c^{(t)}, y_c^{(t)})$ を取得する（6行目）．
取得したデータに対する予測損失 $\ell_c^{(t)}$ を計算し，それをADWIN検出器 $\mathcal{A}_c$ に追加してコンセプトドリフトの監視を行うと同時に，実データは一時的にFIFOバッファ $\mathcal{F}_c$ に保持する（7〜8行目）．

$\mathcal{A}_c$ がドリフトを検知した場合（9行目），検知の根拠となった最適な分割点に基づいて古い概念と新しい概念のデータID集合 $\mathcal{I}_{\text{old}}, \mathcal{I}_{\text{new}}$ を取得する．データIDはここでは時刻 $t$ である．そして，一時保持していた $\mathcal{F}_c$ 内の実データをこれらに基づいて分割し，古い概念のデータ $\mathcal{F}_{c, \text{old}}$ は直前まで使用していたモデルのデータストア $\mathcal{S}_{c, m_c}$ に格納する（11〜12行目）．その後，新しい概念のデータ $\mathcal{F}_{c, \text{new}}$ を用いて後述のサブルーチン \textsc{ResolveDrift} を呼び出し，最適なモデルの切り替え，あるいは新規モデルの作成を行う（13行目）．モデルの選択が完了したのち，新しい概念のデータ $\mathcal{F}_{c, \text{new}}$ は切り替え後の該当データストアに追加され，不要になった古い概念の損失データやバッファの中身を破棄して次の概念の学習に備える（14〜21行目）．
一方，ドリフトが検知されない平時のステップにおいては，$\mathcal{F}_c$ に蓄積されたデータ数が規定のバッファ長 $N_{\text{FIFO}}$ を超えた場合にのみ，最も古いデータを取り出して現在のデータストア $\mathcal{S}_{c, m_c}$ へ追加する（23〜25行目）．

データの取得と振り分けが完了した後，指定された更新間隔 $\tau$ に達したステップでのみモデルの学習を行う．具体的には，データが存在する各データストア $\mathcal{S}_{c, m}$ からミニバッチをサンプリングし，$L \cdot \tau$ 回のローカル更新を行ってワーキング集合内のモデルパラメータを上書きする（26〜30行目）．間隔 $\tau$ に応じて更新回数をスケーリングすることで，毎ステップ更新を行う場合と同等の計算量を維持している．
ラウンドの最終時刻 $rK$ に到達後，更新された既存モデルの集合 $\mathcal{H}_{c, \text{upd}}^{[r]}$ と，新規作成されたモデルの集合 $\mathcal{H}_{c, \text{new}}^{[r]}$ を抽出し，サーバへ送信する（34〜37行目）．

\subsubsection{細かい粒度の処理と統計的検出によるコンセプト混合の防止}
既存のコンセプトドリフト適応の連合学習手法であるFedDrift~\cite{jothimurugesan2023}では，固定サイズのバッチ単位で損失評価・ドリフト検出を行う．しかしこのアプローチでは，ドリフトの発生タイミングがバッチの途中に位置した場合，同一バッチ内に新旧のコンセプトが混在してしまう課題があった．一方，バッチを小さくして検出粒度を細かくすると，単純な損失の増分によるヒューリスティックなドリフト検出により，ノイズへの過敏な反応や計算量の増加を引き起こす可能性があった．本アルゴリズムでは，ADWINを用いてデータストリームを1件ずつ処理することで，細かい粒度での検出が可能である．加えて，信頼度パラメータ $\delta_{\text{adwin}}$ を用いた統計的なドリフト判定を行うため，データのノイズと真のコンセプト変化を統計的に区別できる利点がある．これにより，検出の粒度と精度の両立が可能となった.

\subsubsection{ADWINと遅延バッファを統合したデータ割り当て}
コンセプトドリフトが実際に発生してからADWINがそれを有意な変化として検知するまでには遅延（Detection Delay）が生じる．もし取得したデータを即座にデータストアへ格納してしまうと，この遅延期間中に到着した「新しい概念のデータ」が「古い概念のデータストア」に混入し，学習データの不整合とモデル精度の低下を引き起こす．
この課題に対し，本アルゴリズムでは取得したデータを即座にデータストアに格納せず，FIFOバッファ $\mathcal{F}_c$ （バッファ長 $N_{\text{FIFO}}$）を用いて一時的にデータをプールすることで，意図的な遅延を持たせている．そしてドリフトが検知された瞬間に，ADWINが検出した分割点を利用し，バッファ内の実データを事後的に古い概念と新しい概念のデータ集合へ切り分ける．
このように，ADWINと遅延バッファを統合し，データストアへの厳密なルーティングを行う点に本手法の新規性がある．これにより，データストア内の純度を高く保ちつつ，新規モデルに対して新しい概念のデータのみを供給して学習を行うことが可能となり，過去のデータの保持と新たな概念への柔軟な適応を効果的に両立させている．

\subsection{ドリフト解決処理}

本節では，Algorithm \ref{alg:client_main} でADWINによるドリフト検知後に呼び出されるサブルーチン $\textsc{ResolveDrift}$ について説明する．この処理では，FIFOバッファ $\mathcal{F}_c$ の分割によって得られた新しい概念のデータ群 $\mathcal{F}_{c, \text{new}}$ を用いて，現在の環境に最も適合する既存モデルへの回帰，あるいは新規モデルの作成を行う．モデルの選択にあたっては，$\mathcal{F}_{c, \text{new}}$ に対する各モデル $h_m$ の平均損失 $\bar{L}_{\text{new}}^{(m)}$ と，過去の全期間における平均損失 $\bar{L}_{\text{base}}^{(m)}$ の比較を行う．このプロセスを Algorithm \ref{alg:drift_resolution} として示す．

\begin{algorithm}[htbp]
\caption{$\textsc{ResolveDrift}$ at client $c$ at time $t$}
\label{alg:drift_resolution}
\begin{algorithmic}[1]
\small
\REQUIRE 
    現在のワーキングモデル集合 $\mathcal{H}_{c, \text{work}}^{(t)}$, 新しい概念のデータ群 $\mathcal{F}_{c, \text{new}}$
\PARAMETERS 
    距離閾値 $\gamma_{\text{dist}}$
\ENSURE 
    新しいモデルID $m^*$

\STATE 適合候補のモデルID集合 $\mathcal{M}_{\text{cand}} \gets \emptyset$

\FOR{各モデル $h_m \in \mathcal{H}_{c, \text{work}}^{(t)}$}
    \STATE $\mathcal{F}_{c, \text{new}}$ に対する平均損失 $\bar{L}_{\text{new}}^{(m)} \gets \frac{1}{|\mathcal{F}_{c, \text{new}}|} \sum_{(x,y) \in \mathcal{F}_{c, \text{new}}} \ell(h_m(x), y)$ を計算
    \STATE モデル $h_m$ のベースライン損失 $\bar{L}_{\text{base}}^{(m)}$ を取得
    
    \STATE \COMMENT{ベースラインが存在し，かつ誤差の増加が閾値以内の場合のみ候補とする}
    \IF{$0 < \bar{L}_{\text{base}}^{(m)} \ \AND \ \bar{L}_{\text{new}}^{(m)} - \bar{L}_{\text{base}}^{(m)} \le \gamma_{\text{dist}}$}
        \STATE $\mathcal{M}_{\text{cand}} \gets \mathcal{M}_{\text{cand}} \cup \{m\}$
    \ENDIF
\ENDFOR

\IF{$\mathcal{M}_{\text{cand}} \neq \emptyset$}
    \STATE \COMMENT{既知のコンセプト：損失が最小となる既存モデルを選択}
    \STATE $m^* \gets \argmin\limits_{m \in \mathcal{M}_{\text{cand}}} \bar{L}_{\text{new}}^{(m)}$
\ELSE
    \STATE \COMMENT{未知のコンセプト：新規モデルIDを発行}
    \STATE $m^* \gets \text{新規のモデルID}$
\ENDIF

\RETURN $m^*$
\end{algorithmic}
\end{algorithm}

Algorithm \ref{alg:drift_resolution} は，ドリフト直後の新しいデータに基づいて最適なモデルID $m^*$ を決定する．
アルゴリズム内で評価の基準となるベースライン損失 $\bar{L}_{\text{base}}^{(m)}$ は，サーバでモデルの更新情報を集約する際に，モデルごとに各クライアントから平均損失の情報も集約することで事前に計算・共有されるものである．
アルゴリズムのループ処理（4〜12行目）では，$\mathcal{F}_{c, \text{new}}$ に対する各既存モデルの平均損失 $\bar{L}_{\text{new}}^{(m)}$ を計算し，以下の不等式を満たすかを各モデルにおいて確認する．

\begin{equation}
\label{eq:L_drift}
\bar{L}_{\text{new}}^{(m)} \le \bar{L}_{\text{base}}^{(m)} + \gamma_{\text{dist}}
\end{equation}

この条件式は，あるモデルが新しい概念のデータに対し，過去のパフォーマンス（ベースライン損失）から距離閾値 $\gamma_{\text{dist}}$ 以上悪化していないかの確認を意味する．
各モデルの評価後，以下の基準に従って最終的なモデルID $m^*$ を決定する．
\begin{itemize}
    \item \textbf{既知のコンセプト:} 式(\ref{eq:L_drift})を満たす適合候補モデルが存在する場合（$\mathcal{M}_{\text{cand}} \neq \emptyset$），その中で最も損失が小さいものを現在の環境に適合するモデルとして選択する．
    \item \textbf{未知のコンセプト:} どのモデルも式(\ref{eq:L_drift})を満たさない場合（$\mathcal{M}_{\text{cand}} = \emptyset$），未知のコンセプトが出現したと判断し，新規モデルを作成するための新しいIDを発行する．
\end{itemize}
このように式(\ref{eq:L_drift})に基づくパフォーマンスで評価することで，あるクライアントが自身では経験していないコンセプトであっても，システム全体で獲得したモデルの性能基準に応じた適合判定が可能となる．

\end{document}