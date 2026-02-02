---
title: SD Prompt Tagger
emoji: 🏷️
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 6.5.1
app_file: app.py
pinned: false
license: mit
---

# SD Prompt Tagger

画像からStable Diffusion用のプロンプトタグを自動生成するWebアプリ

## 機能

- **単一画像解析**: 画像をアップロードしてタグを自動生成
- **カテゴリ別分類**: アングル、フォーカス、表情、ポーズ、背景、服装、キャラクター属性
- **BREAK形式出力**: SD用のBREAK区切りプロンプト生成
- **ワイルドカード作成**: 複数画像からカテゴリ別ワイルドカードを生成

## 使用モデル

- [SmilingWolf/wd-eva02-large-tagger-v3](https://huggingface.co/SmilingWolf/wd-eva02-large-tagger-v3)
