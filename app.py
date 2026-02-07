#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SD Prompt Tagger - Web版 (Gradio)
画像からStable Diffusion用のプロンプトタグを自動生成

使用モデル: wd-eva02-large-tagger-v3
"""

import os
import csv
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Optional, List, Dict, Set, Tuple

import numpy as np
from PIL import Image
import gradio as gr

# ONNXランタイム
import onnxruntime as ort

# Hugging Face Hub（モデルダウンロード用）
from huggingface_hub import hf_hub_download


# ============================================================
# 設定
# ============================================================
MODEL_REPO = "SmilingWolf/wd-eva02-large-tagger-v3"
MODEL_FILE = "model.onnx"
TAGS_FILE = "selected_tags.csv"

# デフォルトのしきい値
DEFAULT_GENERAL_THRESHOLD = 0.35
DEFAULT_CHARACTER_THRESHOLD = 0.85

# 品質タグテンプレート（固定） - 廃止予定だが念のため残すか、空にする
QUALITY_TEMPLATE = ""

# タグカテゴリ分類パターン
TAG_CATEGORIES = {
    "quality": {
        "patterns": [
            r"^masterpiece$", r"^best[\s_]quality$", r"^high[\s_]quality$",
            r"^absurdres$", r"^highres$", r"^detailed$", r"^insanely[\s_]detailed$",
            r"^beautiful$", r"^smooth[\s_]quality$", r"^4k$", r"^8k$", r"^aesthetic$",
            r"^very[\s_]aesthetic.*", r"^extremely[\s_]smooth[\s_]skin$", r"^cute[\s_]face$",
            r"^soft[\s_]breasts$",
        ]
    },
    "character_count": {
        "patterns": [
            r"^1girl$", r"^2girls$", r"^3girls$", r"^4girls$", r"^5girls$", r"^6\+girls$", r"^multiple[\s_]girls$",
            r"^1boy$", r"^2boys$", r"^3boys$", r"^4boys$", r"^5boys$", r"^6\+boys$", r"^multiple[\s_]boys$",
            r"^solo$",
        ]
    },
    "hair_color": {
        "patterns": [
            r"^(blonde|black|brown|white|grey|gray|silver|red|blue|green|pink|purple|orange|multicolored|two[_-]?tone|light[\s_]brown|dark[\s_]brown|aqua|cyan|gradient)[\s_]hair$",
        ]
    },
    "hair_style": {
        "patterns": [
            r"^(long|short|medium|very[\s_]long)[\s_]hair$",
            r"^ponytail$", r"^twintails$", r"^twin[\s_]braids$", r"^side[\s_]ponytail$",
            r"^braid$", r"^braids$", r"^side[\s_]braid$", r"^french[\s_]braid$",
            r"^bun$", r"^double[\s_]bun$", r"^hair[\s_]bun$",
            r"^bob[\s_]cut$", r"^hime[\s_]cut$", r"^pixie[\s_]cut$",
            r"^drill[\s_]hair$", r"^curly[\s_]hair$", r"^wavy[\s_]hair$", r"^straight[\s_]hair$",
            r"^messy[\s_]hair$", r"^wet[\s_]hair$", r"^floating[\s_]hair$",
            r"^ahoge$", r"^antenna[\s_]hair$", r"^hair[\s_]intakes$",
            r"^hair[\s_]over.*", r"^bangs$", r"^blunt[\s_]bangs$", r"^swept[\s_]bangs$", r"^parted[\s_]bangs$",
            r"^low[\s_]ponytail$", r"^high[\s_]ponytail$",
            r"^hair[\s_]between[\s_]eyes$", r"^sidelocks$",
        ]
    },
    "eye_color": {
        "patterns": [
            r"^(blue|red|green|brown|yellow|purple|pink|orange|black|white|grey|gray|heterochromia|multicolored|aqua|golden|amber)[\s_]eyes$",
        ]
    },
    "eye_feature": {
        "patterns": [
            r"^detailed[\s_]eyes$", r"^sparkling[\s_]eyes$", r"^glowing[\s_]eyes$",
            r"^half[\s_]closed[\s_]eyes$", r"^closed[\s_]eyes$", r"^empty[\s_]eyes$",
        ]
    },
    "body": {
        "patterns": [
            r"^(small|medium|large|huge|gigantic)[\s_]breasts$",
            r"^flat[\s_]chest$", r"^breasts$",
            r"^wide[\s_]hips$", r"^narrow[\s_]waist$", r"^thick[\s_]thighs$", r"^thighs$",
            r"^ass$", r"^butt$", r"^navel$", r"^collarbone$", r"^midriff$",
            r"^cleavage$", r"^sideboob$", r"^underboob$",
            r"^legs$", r"^kneepits$", r"^armpits$",
        ]
    },
    "angle": {
        "patterns": [
            r"^from[\s_]behind$", r"^from[\s_]above$", r"^from[\s_]below$", r"^from[\s_]side$",
            r"^dutch[\s_]angle$", r"^tilted[\s_]frame$",
            r"^pov$", r"^first[\s_]person[\s_]view$",
            r"^upskirt$", r"^pantyshot$",
            r"^side[\s_]view$", r"^back[\s_]view$", r"^front[\s_]view$",
            r"^three[\s_]quarter[\s_]view$",
        ]
    },
    "focus": {
        "patterns": [
            r"^cowboy[\s_]shot$", r"^upper[\s_]body$", r"^full[\s_]body$", r"^lower[\s_]body$",
            r"^portrait$", r"^close[\s_]up$", r"^face[\s_]focus$", r"^breast[\s_]focus$", r"^ass[\s_]focus$",
            r"^feet[\s_]focus$", r"^eye[\s_]focus$", r"^hip[\s_]focus$",
            r"^head[\s_]focus$", r"^hand[\s_]focus$", r"^leg[\s_]focus$",
            r"^torso$", r"^headshot$",
        ]
    },
    "expression": {
        "patterns": [
            r"^blush$", r"^smile$", r"^grin$", r"^smirk$",
            r"^open[\s_]mouth$", r"^closed[\s_]mouth$", r"^parted[\s_]lips$",
            r"^tongue$", r"^tongue[\s_]out$", r"^licking[\s_]lips$",
            r"^crying$", r"^tears$", r"^teardrop$",
            r"^angry$", r"^annoyed$", r"^serious$", r"^frown$",
            r"^surprised$", r"^shocked$", r"^scared$",
            r"^embarrassed$", r"^shy$", r"^nervous$",
            r"^happy$", r"^excited$", r"^sad$",
            r"^expressionless$", r"^emotionless$",
            r"^ahegao$", r"^orgasm$", r"^fucked[\s_]silly$",
            r"^drooling$", r"^saliva$", r"^sweat$", r"^sweatdrop$",
            r"^light[\s_]smile$", r"^evil[\s_]smile$", r"^seductive[\s_]smile$",
            r"^pout$", r"^furrowed[\s_]brow$",
        ]
    },
    "pose": {
        "patterns": [
            r"^standing$", r"^sitting$", r"^lying$", r"^kneeling$", r"^squatting$", r"^crouching$",
            r"^walking$", r"^running$", r"^jumping$",
            r"^bent[\s_]over$", r"^leaning[\s_]forward$", r"^leaning[\s_]back$",
            r"^crossed[\s_]arms$", r"^crossed[\s_]legs$",
            r"^arms[\s_]up$", r"^arms[\s_]behind[\s_]back$", r"^arms[\s_]behind[\s_]head$",
            r"^hand[\s_]on[\s_]hip$", r"^hands[\s_]on[\s_]hips$",
            r"^hands[\s_]on[\s_]own[\s_]knees$", r"^hands[\s_]on[\s_]own[\s_]chest$",
            r"^looking[\s_]at[\s_]viewer$", r"^looking[\s_]away$", r"^looking[\s_]back$",
            r"^looking[\s_]up$", r"^looking[\s_]down$", r"^looking[\s_]to[\s_]the[\s_]side$",
            r"^on[\s_]bed$", r"^on[\s_]floor$", r"^on[\s_]stomach$", r"^on[\s_]back$",
            r"^spread[\s_]legs$", r"^legs[\s_]together$", r"^legs[\s_]apart$",
            r"^breast[\s_]rest$", r"^breasts[\s_]on[\s_]table$",
            r"^stretching$", r"^reaching$", r"^pointing$",
            r"^waving$", r"^peace[\s_]sign$", r"^v[\s_]sign$",
        ]
    },
    "background": {
        "patterns": [
            r".*background$",
            r"^outdoors$", r"^indoors$",
            r"^sky$", r"^cloud$", r"^clouds$", r"^starry[\s_]sky$",
            r"^grass$", r"^tree$", r"^trees$", r"^water$", r"^ocean$", r"^beach$",
            r"^night$", r"^day$", r"^sunset$", r"^sunrise$", r"^dusk$", r"^dawn$",
            r"^forest$", r"^city$", r"^cityscape$", r"^building$", r"^buildings$",
            r"^classroom$", r"^bedroom$", r"^bathroom$", r"^kitchen$", r"^living[\s_]room$",
            r"^school$", r"^office$", r"^hospital$", r"^church$",
            r"^window$", r"^door$", r"^curtains$", r"^wall$", r"^floor$",
            r"^desk$", r"^table$", r"^chair$", r"^bed$", r"^sofa$", r"^couch$",
            r"^school[\s_]desk$", r"^school[\s_]chair$",
            r"^street$", r"^road$", r"^path$", r"^bridge$",
            r"^garden$", r"^park$", r"^rooftop$", r"^balcony$",
            r"^train$", r"^bus$", r"^car$", r"^vehicle[\s_]interior$",
            r"^book$", r"^books$", r"^bookshelf$",
        ]
    },
    "clothing": {
        "patterns": [
            # 上半身
            r"^shirt$", r"^blouse$", r"^t[\s_-]?shirt$", r"^tank[\s_]top$",
            r"^sweater$", r"^cardigan$", r"^hoodie$", r"^jacket$", r"^coat$",
            r"^vest$", r"^crop[\s_]top$", r"^tube[\s_]top$",
            r"^sailor[\s_]collar$", r"^collared[\s_]shirt$",
            # 下半身
            r"^skirt$", r"^miniskirt$", r"^long[\s_]skirt$", r"^pleated[\s_]skirt$",
            r"^pants$", r"^jeans$", r"^shorts$", r"^short[\s_]shorts$",
            r"^leggings$", r"^sweatpants$",
            # ドレス・ワンピース
            r"^dress$", r"^sundress$", r"^wedding[\s_]dress$", r"^evening[\s_]dress$",
            r"^cocktail[\s_]dress$", r"^maid[\s_]dress$", r"^china[\s_]dress$",
            # 制服
            r"^uniform$", r"^school[\s_]uniform$", r"^serafuku$", r"^sailor[\s_]uniform$",
            r"^military[\s_]uniform$", r"^maid[\s_]uniform$", r"^nurse[\s_]uniform$",
            r"^police[\s_]uniform$", r"^cheerleader$",
            # 水着・下着
            r"^swimsuit$", r"^bikini$", r"^one[\s_-]?piece[\s_]swimsuit$",
            r"^school[\s_]swimsuit$", r"^competition[\s_]swimsuit$",
            r"^underwear$", r"^bra$", r"^panties$", r"^lingerie$",
            r"^thong$", r"^g[\s_-]?string$",
            # レッグウェア
            r"^thighhighs$", r"^stockings$", r"^pantyhose$", r"^socks$",
            r"^knee[\s_]highs$", r"^ankle[\s_]socks$", r"^legwear$",
            r"^garter[\s_]belt$", r"^garter[\s_]straps$",
            # 靴
            r"^shoes$", r"^boots$", r"^high[\s_]heels$", r"^sandals$",
            r"^sneakers$", r"^loafers$", r"^mary[\s_]janes$",
            r"^knee[\s_]boots$", r"^thigh[\s_]boots$",
            # アクセサリー・小物
            r"^gloves$", r"^fingerless[\s_]gloves$", r"^elbow[\s_]gloves$",
            r"^hat$", r"^cap$", r"^beret$", r"^ribbon$", r"^bow$",
            r"^scarf$", r"^necktie$", r"^bowtie$", r"^choker$", r"^collar$",
            r"^glasses$", r"^sunglasses$", r"^eyepatch$",
            r"^earrings$", r"^necklace$", r"^bracelet$", r"^ring$",
            r"^hairband$", r"^headband$", r"^hair[\s_]ribbon$", r"^hair[\s_]bow$",
            r"^hair[\s_]ornament$", r"^hairclip$", r"^hairpin$",
            # 和服
            r"^kimono$", r"^yukata$", r"^hakama$", r"^miko$",
            # その他
            r"^apron$", r"^cape$", r"^cloak$", r"^robe$",
            r"^armor$", r"^bodysuit$", r"^leotard$", r"^catsuit$",
            r"^pajamas$", r"^nightgown$",
        ]
    },
}


# ============================================================
# モデル管理
# ============================================================
class WDTagger:
    """WD Taggerモデルクラス"""

    def __init__(self, model_dir: Optional[Path] = None):
        self.model_dir = model_dir or Path(__file__).parent / "models" / "wd-eva02-large-tagger-v3"
        self.session: Optional[ort.InferenceSession] = None
        self.tags: list = []
        self.general_tags: list = []
        self.character_tags: list = []
        self.rating_tags: list = []

    def ensure_model(self) -> bool:
        """モデルファイルが存在しない場合はダウンロード"""
        self.model_dir.mkdir(parents=True, exist_ok=True)

        model_path = self.model_dir / MODEL_FILE
        tags_path = self.model_dir / TAGS_FILE

        files_to_download = []
        if not model_path.exists():
            files_to_download.append((MODEL_FILE, model_path))
        if not tags_path.exists():
            files_to_download.append((TAGS_FILE, tags_path))

        if files_to_download:
            for filename, local_path in files_to_download:
                print(f"ダウンロード中: {filename}")
                hf_hub_download(
                    repo_id=MODEL_REPO,
                    filename=filename,
                    local_dir=self.model_dir,
                    local_dir_use_symlinks=False
                )

        return True

    def load(self) -> bool:
        """モデルを読み込み"""
        print("モデルを読み込み中...")

        model_path = self.model_dir / MODEL_FILE
        tags_path = self.model_dir / TAGS_FILE

        # ONNXセッションを作成
        providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
        available_providers = ort.get_available_providers()
        providers = [p for p in providers if p in available_providers]

        self.session = ort.InferenceSession(str(model_path), providers=providers)

        # タグを読み込み
        with open(tags_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                tag_name = row['name']
                category = int(row['category'])
                self.tags.append(tag_name)

                if category == 0:
                    self.general_tags.append(tag_name)
                elif category == 4:
                    self.character_tags.append(tag_name)
                elif category == 9:
                    self.rating_tags.append(tag_name)

        print("モデル読み込み完了")
        return True

    def predict(
        self,
        image: Image.Image,
        general_threshold: float = DEFAULT_GENERAL_THRESHOLD,
        character_threshold: float = DEFAULT_CHARACTER_THRESHOLD,
    ) -> dict:
        """画像からタグを予測"""
        if self.session is None:
            raise RuntimeError("モデルが読み込まれていません")

        # 画像の前処理
        input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        height, width = input_shape[1], input_shape[2]

        # リサイズとパディング
        image = image.convert('RGB')

        img_ratio = image.width / image.height
        target_ratio = width / height

        if img_ratio > target_ratio:
            new_width = width
            new_height = int(width / img_ratio)
        else:
            new_height = height
            new_width = int(height * img_ratio)

        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # パディング（白で埋める）
        padded = Image.new('RGB', (width, height), (255, 255, 255))
        paste_x = (width - new_width) // 2
        paste_y = (height - new_height) // 2
        padded.paste(image, (paste_x, paste_y))

        # NumPy配列に変換
        img_array = np.array(padded, dtype=np.float32)
        img_array = img_array[:, :, ::-1]  # RGB to BGR
        img_array = np.expand_dims(img_array, axis=0)

        # 推論
        outputs = self.session.run(None, {input_name: img_array})
        probs = outputs[0][0]

        # 結果を整理
        results = {
            'general': [],
            'character': [],
            'rating': {},
            'all_tags': [],
        }

        for idx, (tag, prob) in enumerate(zip(self.tags, probs)):
            prob_float = float(prob)

            if tag in self.rating_tags:
                results['rating'][tag] = prob_float
            elif tag in self.character_tags:
                if prob_float >= character_threshold:
                    results['character'].append((tag, prob_float))
            elif tag in self.general_tags:
                if prob_float >= general_threshold:
                    results['general'].append((tag, prob_float))

        results['general'].sort(key=lambda x: x[1], reverse=True)
        results['character'].sort(key=lambda x: x[1], reverse=True)

        all_tags = []
        all_tags.extend([tag for tag, _ in results['character']])
        all_tags.extend([tag for tag, _ in results['general']])
        results['all_tags'] = all_tags

        return results


# ============================================================
# タグフォーマッター
# ============================================================
class TagFormatter:
    """タグのフォーマット処理"""

    @staticmethod
    def categorize_tag(tag: str) -> str:
        """タグをカテゴリに分類"""
        tag_lower = tag.lower()
        tag_normalized = tag_lower.replace('_', ' ')

        for category, info in TAG_CATEGORIES.items():
            patterns = info.get("patterns", [])
            for pattern in patterns:
                if re.match(pattern, tag_lower, re.IGNORECASE) or re.match(pattern, tag_normalized, re.IGNORECASE):
                    return category

        return "other"

    @staticmethod
    def categorize_all_tags(tags: List[str]) -> Dict[str, List[str]]:
        """全タグをカテゴリ別に分類"""
        categorized = {
            "angle": [],
            "focus": [],
            "expression": [],
            "pose": [],
            "background": [],
            "clothing": [],
            "character_attr": [],
            "other": []
        }

        character_categories = {"character_count", "hair_color", "hair_style", "eye_color", "eye_feature", "body"}

        for tag in tags:
            category = TagFormatter.categorize_tag(tag)
            formatted_tag = tag.replace('_', ' ')

            if category in ["angle", "focus", "expression", "pose", "background", "clothing"]:
                categorized[category].append(formatted_tag)
            elif category in character_categories:
                categorized["character_attr"].append(formatted_tag)
            else:
                categorized["other"].append(formatted_tag)

        return categorized

    @staticmethod
    def format_tags(tags: List[str], replace_underscore: bool = True) -> str:
        """タグリストをフォーマットして文字列に変換"""
        formatted = []
        for tag in tags:
            t = tag.replace('_', ' ') if replace_underscore else tag
            formatted.append(t)
        return ", ".join(formatted)

    @staticmethod
    def format_with_break(tags: List[str], quality_template: str = "") -> str:
        """タグをカテゴリ別に分類してBREAK形式で出力"""
        categorized = {"quality": [], "character_attr": [], "pose_angle": []}

        character_categories = {"character_count", "hair_color", "hair_style", "eye_color", "eye_feature", "body"}
        pose_categories = {"angle", "focus", "expression", "pose", "background"}

        for tag in tags:
            category = TagFormatter.categorize_tag(tag)
            formatted_tag = tag.replace('_', ' ')

            if category == "quality":
                categorized["quality"].append(formatted_tag)
            elif category in character_categories:
                categorized["character_attr"].append(formatted_tag)
            elif category in pose_categories:
                categorized["pose_angle"].append(formatted_tag)
            else:
                categorized["pose_angle"].append(formatted_tag)

        sections = []
        
        # Quality Section (User Input + Detected)
        quality_section = []
        if quality_template:
             quality_section.extend([t.strip() for t in quality_template.split(',') if t.strip()])
        
        # Deduplicate detected tags against user input if needed, or just append. 
        # Plan says "Ensure tags are not duplicated".
        for qt in categorized["quality"]:
            if qt not in quality_section:
                quality_section.append(qt)
        
        if quality_section:
            sections.append(", ".join(quality_section) + ",")

        if categorized["character_attr"]:
            sections.append(", ".join(categorized["character_attr"]) + ",")

        if categorized["pose_angle"]:
            sections.append(", ".join(categorized["pose_angle"]) + ",")

        return "\nBREAK\n".join(sections)


# ============================================================
# グローバルインスタンス
# ============================================================
tagger = WDTagger()


# ============================================================
# Gradio関数
# ============================================================
def analyze_single_image(
    image: Image.Image,
    general_threshold: float,
    character_threshold: float,
    use_break_format: bool,
    quality_tags: str = "",
) -> Tuple[str, str, str, str, str, str, str, str, str]:
    """単一画像を解析"""
    if image is None:
        return "", "", "", "", "", "", "", "", ""

    results = tagger.predict(image, general_threshold, character_threshold)

    # タグをフォーマット
    if use_break_format:
        main_tags = TagFormatter.format_with_break(results['all_tags'], quality_template=quality_tags)
    else:
        main_tags = TagFormatter.format_tags(results['all_tags'])

    # カテゴリ別に分類
    categorized = TagFormatter.categorize_all_tags(results['all_tags'])

    # Rating情報
    rating_info = ""
    for rating, prob in sorted(results['rating'].items(), key=lambda x: x[1], reverse=True):
        rating_info += f"{rating}: {prob:.2%}\n"

    return (
        main_tags,
        ", ".join(categorized["angle"]),
        ", ".join(categorized["focus"]),
        ", ".join(categorized["expression"]),
        ", ".join(categorized["pose"]),
        ", ".join(categorized["background"]),
        ", ".join(categorized["clothing"]),
        ", ".join(categorized["character_attr"]),
        rating_info
    )


def create_wildcard_file(tags_text: str, category_name: str) -> Optional[str]:
    """単一カテゴリのワイルドカードファイルを作成"""
    if not tags_text or not tags_text.strip():
        return None

    # カンマ区切りを改行区切りに変換
    tags = [t.strip() for t in tags_text.split(",") if t.strip()]
    if not tags:
        return None

    # 一時ファイルを作成
    tmp_dir = tempfile.mkdtemp()
    filepath = os.path.join(tmp_dir, f"{category_name}.txt")
    with open(filepath, 'w', encoding='utf-8') as f:
        for tag in tags:
            f.write(f"{tag}\n")

    return filepath


def create_all_wildcards_zip(
    angle: str, focus: str, expression: str, pose: str,
    background: str, clothing: str, char_attr: str
) -> Optional[str]:
    """全カテゴリのワイルドカードをZIPファイルとして作成"""
    categories = {
        "angle": angle,
        "focus": focus,
        "expression": expression,
        "pose": pose,
        "background": background,
        "clothing": clothing,
        "character": char_attr,
    }

    # 空でないカテゴリがあるかチェック
    has_content = any(v and v.strip() for v in categories.values())
    if not has_content:
        return None

    # 一時ディレクトリにZIPファイルを作成
    tmp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(tmp_dir, "wildcards.zip")

    with zipfile.ZipFile(zip_path, 'w') as zf:
        for cat_name, tags_text in categories.items():
            if tags_text and tags_text.strip():
                tags = [t.strip() for t in tags_text.split(",") if t.strip()]
                if tags:
                    content = "\n".join(tags)
                    zf.writestr(f"{cat_name}.txt", content)

    return zip_path


def create_selected_wildcards_zip(
    angle: str, focus: str, expression: str, pose: str,
    background: str, clothing: str, char_attr: str,
    chk_angle: bool, chk_focus: bool, chk_expression: bool, chk_pose: bool,
    chk_background: bool, chk_clothing: bool, chk_char_attr: bool
) -> Optional[str]:
    """選択されたカテゴリのワイルドカードをZIPファイルとして作成"""
    categories = {
        "angle": (angle, chk_angle),
        "focus": (focus, chk_focus),
        "expression": (expression, chk_expression),
        "pose": (pose, chk_pose),
        "background": (background, chk_background),
        "clothing": (clothing, chk_clothing),
        "character": (char_attr, chk_char_attr),
    }

    # 選択されたカテゴリで空でないものがあるかチェック
    has_content = any(
        checked and tags_text and tags_text.strip()
        for tags_text, checked in categories.values()
    )
    if not has_content:
        return None

    # 一時ディレクトリにZIPファイルを作成
    tmp_dir = tempfile.mkdtemp()
    zip_path = os.path.join(tmp_dir, "wildcards.zip")

    with zipfile.ZipFile(zip_path, 'w') as zf:
        for cat_name, (tags_text, checked) in categories.items():
            if checked and tags_text and tags_text.strip():
                tags = [t.strip() for t in tags_text.split(",") if t.strip()]
                if tags:
                    content = "\n".join(tags)
                    zf.writestr(f"{cat_name}.txt", content)

    return zip_path


def analyze_multiple_images(
    images: List[Image.Image],
    general_threshold: float,
    character_threshold: float,
    progress=gr.Progress()
) -> Tuple[str, str, str, str, str, str, str, str]:
    """複数画像を解析してワイルドカード用に集計"""
    if not images:
        return "", "", "", "", "", "", "", ""

    collected: Dict[str, Set[str]] = {
        "angle": set(),
        "focus": set(),
        "expression": set(),
        "pose": set(),
        "background": set(),
        "clothing": set(),
        "character_attr": set(),
        "other": set(),
    }

    character_categories = {"character_count", "hair_color", "hair_style", "eye_color", "eye_feature", "body"}

    for i, img in enumerate(progress.tqdm(images, desc="処理中")):
        if img is None:
            continue

        results = tagger.predict(img, general_threshold, character_threshold)

        for tag in results['all_tags']:
            category = TagFormatter.categorize_tag(tag)
            formatted_tag = tag.replace('_', ' ')

            if category in ["angle", "focus", "expression", "pose", "background", "clothing"]:
                collected[category].add(formatted_tag)
            elif category in character_categories:
                collected["character_attr"].add(formatted_tag)
            else:
                collected["other"].add(formatted_tag)

    return (
        "\n".join(sorted(collected["angle"])),
        "\n".join(sorted(collected["focus"])),
        "\n".join(sorted(collected["expression"])),
        "\n".join(sorted(collected["pose"])),
        "\n".join(sorted(collected["background"])),
        "\n".join(sorted(collected["clothing"])),
        "\n".join(sorted(collected["character_attr"])),
        "\n".join(sorted(collected["other"])),
    )


# ============================================================
# Gradio UI
# ============================================================
def create_ui():
    """Gradio UIを構築"""

    with gr.Blocks(title="SD Prompt Tagger") as app:
        gr.Markdown("# SD Prompt Tagger")
        gr.Markdown("画像からStable Diffusion用のプロンプトタグを自動生成します")

        with gr.Tabs():
            # === タブ1: 単一画像解析 ===
            with gr.TabItem("単一画像"):
                with gr.Row():
                    with gr.Column(scale=1):
                        single_image = gr.Image(type="pil", label="画像をアップロード")

                        with gr.Group():
                            gr.Markdown("### しきい値設定")
                            general_threshold = gr.Slider(
                                minimum=0.0, maximum=1.0, value=0.35, step=0.05,
                                label="一般タグ"
                            )
                            char_threshold = gr.Slider(
                                minimum=0.0, maximum=1.0, value=0.85, step=0.05,
                                label="キャラクタータグ"
                            )

                        with gr.Row():
                            gr.Button("バランス", size="sm").click(
                                fn=lambda: (0.35, 0.85),
                                outputs=[general_threshold, char_threshold]
                            )
                            gr.Button("タグ多め", size="sm").click(
                                fn=lambda: (0.25, 0.75),
                                outputs=[general_threshold, char_threshold]
                            )
                            gr.Button("厳選", size="sm").click(
                                fn=lambda: (0.50, 0.90),
                                outputs=[general_threshold, char_threshold]
                            )

                        quality_tags_input = gr.Textbox(
                            label="追加の品質タグ（検出された品質タグの前に手動で追加）",
                            lines=2,
                            value="",
                            placeholder="必要であれば追加の品質タグを入力..."
                        )

                        use_break = gr.Checkbox(label="BREAK形式で出力", value=True)
                        analyze_btn = gr.Button("タグを生成", variant="primary")

                    with gr.Column(scale=2):
                        main_output = gr.Textbox(label="生成されたタグ", lines=5)

                        gr.Markdown("### カテゴリ別タグ")
                        with gr.Row():
                            angle_output = gr.Textbox(label="アングル", lines=2)
                            focus_output = gr.Textbox(label="フォーカス", lines=2)
                            expression_output = gr.Textbox(label="表情", lines=2)

                        with gr.Row():
                            pose_output = gr.Textbox(label="ポーズ", lines=2)
                            background_output = gr.Textbox(label="背景", lines=2)
                            clothing_output = gr.Textbox(label="服装", lines=2)

                        with gr.Row():
                            char_attr_output = gr.Textbox(label="キャラクター属性", lines=2)
                            rating_output = gr.Textbox(label="Rating", lines=2)

                        gr.Markdown("### ワイルドカード出力")
                        gr.Markdown("ダウンロードするカテゴリを選択:")
                        with gr.Row():
                            chk_angle = gr.Checkbox(label="アングル", value=True)
                            chk_focus = gr.Checkbox(label="フォーカス", value=True)
                            chk_expression = gr.Checkbox(label="表情", value=True)
                            chk_pose = gr.Checkbox(label="ポーズ", value=True)
                        with gr.Row():
                            chk_background = gr.Checkbox(label="背景", value=True)
                            chk_clothing = gr.Checkbox(label="服装", value=True)
                            chk_char_attr = gr.Checkbox(label="キャラクター属性", value=True)

                        with gr.Row():
                            download_wildcards_btn = gr.Button("ワイルドカードをダウンロード", variant="secondary")
                            wildcard_download = gr.File(label="ダウンロード", visible=False)

                analyze_btn.click(
                    fn=analyze_single_image,
                    inputs=[single_image, general_threshold, char_threshold, use_break, quality_tags_input],
                    outputs=[main_output, angle_output, focus_output, expression_output,
                             pose_output, background_output, clothing_output, char_attr_output, rating_output]
                )

                # 画像アップロード時も自動解析
                single_image.change(
                    fn=analyze_single_image,
                    inputs=[single_image, general_threshold, char_threshold, use_break, quality_tags_input],
                    outputs=[main_output, angle_output, focus_output, expression_output,
                             pose_output, background_output, clothing_output, char_attr_output, rating_output]
                )

                # ワイルドカードダウンロード（カテゴリ選択付き）
                download_wildcards_btn.click(
                    fn=create_selected_wildcards_zip,
                    inputs=[angle_output, focus_output, expression_output, pose_output,
                            background_output, clothing_output, char_attr_output,
                            chk_angle, chk_focus, chk_expression, chk_pose,
                            chk_background, chk_clothing, chk_char_attr],
                    outputs=[wildcard_download]
                ).then(
                    fn=lambda x: gr.update(visible=True) if x else gr.update(visible=False),
                    inputs=[wildcard_download],
                    outputs=[wildcard_download]
                )

            # === タブ2: ワイルドカード作成 ===
            with gr.TabItem("ワイルドカード作成"):
                gr.Markdown("複数の画像からタグを抽出し、カテゴリ別にワイルドカードを作成します")

                with gr.Row():
                    with gr.Column(scale=1):
                        multi_images = gr.Gallery(
                            label="画像をアップロード（複数可）",
                            columns=3,
                            height=300,
                            object_fit="contain"
                        )
                        multi_files = gr.File(
                            label="画像ファイルを選択",
                            file_count="multiple",
                            file_types=["image"]
                        )

                        with gr.Group():
                            gr.Markdown("### しきい値設定")
                            wc_general_threshold = gr.Slider(
                                minimum=0.0, maximum=1.0, value=0.35, step=0.05,
                                label="一般タグ"
                            )
                            wc_char_threshold = gr.Slider(
                                minimum=0.0, maximum=1.0, value=0.85, step=0.05,
                                label="キャラクタータグ"
                            )

                        extract_btn = gr.Button("タグを抽出", variant="primary")

                    with gr.Column(scale=2):
                        gr.Markdown("### カテゴリ別タグ（1行1タグ = ワイルドカード形式）")
                        with gr.Row():
                            wc_angle = gr.Textbox(label="アングル", lines=8)
                            wc_focus = gr.Textbox(label="フォーカス", lines=8)
                            wc_expression = gr.Textbox(label="表情", lines=8)

                        with gr.Row():
                            wc_pose = gr.Textbox(label="ポーズ", lines=8)
                            wc_background = gr.Textbox(label="背景", lines=8)
                            wc_clothing = gr.Textbox(label="服装", lines=8)

                        with gr.Row():
                            wc_char_attr = gr.Textbox(label="キャラクター属性", lines=8)
                            wc_other = gr.Textbox(label="その他", lines=8)

                        gr.Markdown("### ダウンロードするカテゴリを選択")
                        with gr.Row():
                            wc_chk_angle = gr.Checkbox(label="アングル", value=True)
                            wc_chk_focus = gr.Checkbox(label="フォーカス", value=True)
                            wc_chk_expression = gr.Checkbox(label="表情", value=True)
                            wc_chk_pose = gr.Checkbox(label="ポーズ", value=True)
                        with gr.Row():
                            wc_chk_background = gr.Checkbox(label="背景", value=True)
                            wc_chk_clothing = gr.Checkbox(label="服装", value=True)
                            wc_chk_char_attr = gr.Checkbox(label="キャラクター属性", value=True)
                            wc_chk_other = gr.Checkbox(label="その他", value=True)

                        with gr.Row():
                            wc_download_btn = gr.Button("選択カテゴリをZIPでダウンロード", variant="secondary")
                            wc_download_file = gr.File(label="ダウンロード", visible=False)

                # ファイルアップロード時にギャラリーを更新
                def update_gallery(files):
                    if files is None:
                        return []
                    images = []
                    for f in files:
                        try:
                            img = Image.open(f.name)
                            images.append(img)
                        except:
                            pass
                    return images

                multi_files.change(
                    fn=update_gallery,
                    inputs=[multi_files],
                    outputs=[multi_images]
                )

                # タグ抽出
                def extract_from_files(files, general_th, char_th, progress=gr.Progress()):
                    if files is None:
                        return "", "", "", "", "", "", "", ""
                    images = []
                    for f in files:
                        try:
                            img = Image.open(f.name)
                            images.append(img)
                        except:
                            pass
                    return analyze_multiple_images(images, general_th, char_th, progress)

                extract_btn.click(
                    fn=extract_from_files,
                    inputs=[multi_files, wc_general_threshold, wc_char_threshold],
                    outputs=[wc_angle, wc_focus, wc_expression, wc_pose, wc_background, wc_clothing, wc_char_attr, wc_other]
                )

                # ワイルドカードZIPダウンロード（改行区切り用）
                def create_selected_wildcards_zip_from_newline(
                    angle: str, focus: str, expression: str, pose: str,
                    background: str, clothing: str, char_attr: str, other: str,
                    chk_angle: bool, chk_focus: bool, chk_expression: bool, chk_pose: bool,
                    chk_background: bool, chk_clothing: bool, chk_char_attr: bool, chk_other: bool
                ) -> Optional[str]:
                    """選択されたカテゴリの改行区切りテキストからZIPファイルを作成"""
                    categories = {
                        "angle": (angle, chk_angle),
                        "focus": (focus, chk_focus),
                        "expression": (expression, chk_expression),
                        "pose": (pose, chk_pose),
                        "background": (background, chk_background),
                        "clothing": (clothing, chk_clothing),
                        "character": (char_attr, chk_char_attr),
                        "other": (other, chk_other),
                    }

                    has_content = any(
                        checked and tags_text and tags_text.strip()
                        for tags_text, checked in categories.values()
                    )
                    if not has_content:
                        return None

                    tmp_dir = tempfile.mkdtemp()
                    zip_path = os.path.join(tmp_dir, "wildcards.zip")

                    with zipfile.ZipFile(zip_path, 'w') as zf:
                        for cat_name, (tags_text, checked) in categories.items():
                            if checked and tags_text and tags_text.strip():
                                # 既に改行区切りなのでそのまま使用
                                zf.writestr(f"{cat_name}.txt", tags_text.strip())

                    return zip_path

                wc_download_btn.click(
                    fn=create_selected_wildcards_zip_from_newline,
                    inputs=[wc_angle, wc_focus, wc_expression, wc_pose,
                            wc_background, wc_clothing, wc_char_attr, wc_other,
                            wc_chk_angle, wc_chk_focus, wc_chk_expression, wc_chk_pose,
                            wc_chk_background, wc_chk_clothing, wc_chk_char_attr, wc_chk_other],
                    outputs=[wc_download_file]
                ).then(
                    fn=lambda x: gr.update(visible=True) if x else gr.update(visible=False),
                    inputs=[wc_download_file],
                    outputs=[wc_download_file]
                )

    return app


# ============================================================
# メイン
# ============================================================
def main():
    # モデルを読み込み
    tagger.ensure_model()
    tagger.load()

    # Gradioアプリを起動
    app = create_ui()
    app.launch(
        server_name="0.0.0.0",  # ローカルネットワークからもアクセス可能
        server_port=7860,
        share=False,  # 公開リンクを作成しない
        auth=None,  # 認証なし（必要なら ("user", "password") を設定）
    )


if __name__ == "__main__":
    main()
