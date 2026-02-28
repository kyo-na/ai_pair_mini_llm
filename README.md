# 開発過程・学習の記録  
*English version follows below / 英語版は下部に記載*

---

## 本・AI・実装を往復しながらの理解

本ディレクトリの実装に至るまで、Transformer / Attention / 量子化 / ハードウェア実装について、  
**書籍・論文・既存実装・AI ツール**を往復しながら理解を積み重ねるという、非常に試行錯誤の多いプロセスを経ました。

特に以下の点は、

> 読む → 分かったつもり → 実装 → 壊れる → 調べ直す

を何度も繰り返しています。

- Attention が数式上どうなっているか  
- それを逐次・並列・パイプラインでどう分解できるか  
- GPU 的実装と FPGA 的実装の発想の違い  
- 量子化（Ternary / BitNet）が演算器レベルで何を意味するか  

---

## AI を「答え」ではなく「思考補助」として使う

AI ツールは、完成コードを生成させる目的ではなく、以下の用途で利用しました。

- 数式や処理フローの言語化
- 「この構造はハードウェア的に成立するか？」という設計壁打ち
- 実装が破綻した原因の切り分け
- FPGA 観点での設計妥当性の確認

AI の回答をそのまま採用することはほぼなく、  
**必ず書籍・既存実装・コード・挙動と突き合わせて検証**しています。

---

## 理解に最も苦労した点

特に設計・理解に時間を要したのは以下の点です。

- Causal Attention における時間方向依存性
- KV Cache をメモリ構造としてどう扱うか
- Softmax を近似・ROM・パイプラインでどう成立させるか
- GPU 的な巨大並列と FPGA 的な細粒度パイプラインの考え方の違い

理論的な説明は容易でも、  
**「実装として成立させる」ためには何度も壊して考え直す必要がありました。**

---

## Workingset / スケジューリングへの発展

開発途中で Workingset を調査する機会があり、  
**SSD の I/O スパイクにより CPU・GPU が待ち状態になるケース**を観測しました。

この問題に対し、

- スレッド分割
- タスクキュー化
- 実行単位ごとの占有率（CPU/GPU）をもとにした簡易スケジューリング

といった仕組みを追加し、  
**GPU が使えない場合は CPU に回す / 空いていれば即投入する**構造を試験的に実装しています。

※ GPU 実装の詳細については、法的・利用条件の観点から本リポジトリには記載していません。

---

## このディレクトリの位置づけ

本コードは、

- AI に書かせたコードの集合  
ではなく、
- **AI を使い切ることで理解を深めるための途中結果の記録**

として位置づけています。

そのため、

- 完全な最適解ではない
- 冗長な構成を含む
- 実験的・未整理な実装も存在する

一方で、  
**Transformer をハードウェア視点で理解しようとした痕跡はすべて残しています。**

---

## ⚠️ モデルウェイトについて

本リポジトリには **モデルウェイトは含まれていません**。

GGUF 等のモデルファイルは、各ライセンスに従い、  
別途取得のうえ `models/` ディレクトリ（`.gitignore` 対象）に配置してください。

---

## Development Notes (English)

This repository contains source code only for a custom inference runtime.  
No model weights are included.

This implementation was developed through extensive iteration between books, papers, existing implementations, and AI tools.

Understanding was achieved through repeated cycles of:

> read → assume understanding → implement → fail → re-study

especially in areas such as:

- Mathematical structure of Attention
- Pipeline decomposition for hardware implementation
- GPU-oriented vs FPGA-oriented design approaches
- Meaning of quantization at the arithmetic unit level

AI tools were used strictly as thinking partners, not as final code generators.

---

## Disclaimer

This repository is a personal learning and experimental project created for educational and research purposes.

- It is not intended to replicate, replace, or compete with any existing commercial system.
- The implementation is based on publicly available knowledge and independent experimentation.
- No proprietary source code, confidential materials, or model weights are included.

All code reflects the author’s own understanding and design decisions made during the learning process.

The project is provided “as is” and is intended solely as a record of technical exploration.
