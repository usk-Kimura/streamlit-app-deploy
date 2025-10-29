import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
# app.py
# -----------------------------------------
# 単一入力フォーム + ラジオボタンで専門家A/Bを切替
# LangChain経由でLLMにプロンプトを渡し、回答を表示するStreamlitアプリ
# Lesson8スタイル: PromptTemplate -> LLM -> OutputParser のチェーン
# -----------------------------------------

import streamlit as st

# LangChain (OpenAI) - Lesson8相当の基本構成
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# ====== 設定 ======
# ・環境変数 OPENAI_API_KEY をセットしてから実行してください
#   macOS/Linux:  export OPENAI_API_KEY="sk-xxxx"
#   Windows(PowerShell):  setx OPENAI_API_KEY "sk-xxxx"  (再起動後に有効)
#
# ・モデルは必要に応じて変更可（例: "gpt-4o-mini" / "gpt-4.1-mini" など）
MODEL_NAME = "gpt-4o-mini"
TEMPERATURE = 0.2


# ====== LLMインスタンス ======
llm = ChatOpenAI(model=MODEL_NAME, temperature=TEMPERATURE)


# ====== 専門家プロンプト（A/Bで切替） ======
# ここはお好みで増減・調整OK
EXPERT_SYSTEM_PROMPTS = {
    # A: 建設DX・現場最適化の専門家（たかさんに合わせてAは建設寄りに！）
    "A（建設DXコンサル）": """あなたは建設DXと施工管理の専門家です。
現場安全・工程管理・原価低減・サプライチェーン・AI/データ活用に精通しています。
回答は「結論 → 手順 → ツール例 → リスク/注意点 → 次アクション」の順で、現場ですぐ使える実務レベルで簡潔に提案してください。
専門用語は短く補足を入れてください。""",

    # B: TikTokグロースの専門家（先輩の事業にドンピシャなもう一枠）
    "B（TikTokグロース）": """あなたはTikTokのグロースハッカーで、視聴維持とフォロー獲得に特化した専門家です。
行動経済学（プロスペクト理論）・カリギュラ効果・3秒フック・CTA設計・A/Bテストに精通しています。
回答は「結論 → 30秒台本例 → フック/CTA案 → ハッシュタグ方針 → 計測/KPI → 明日やること」で出力してください。
専門用語は短く補足を入れてください。"""
}


# ====== コア関数（指定：引数=入力テキスト＆選択値 / 戻り値=LLM回答） ======
def ask_expert(input_text: str, expert_choice: str) -> str:
    """
    入力テキストと専門家選択値（ラジオボタンの値）を受け取り、
    専門家に応じたSystemメッセージでLLMへ投げ、文字列として回答を返す。
    """
    if not input_text or not input_text.strip():
        return "入力テキストが空です。内容を入力してください。"

    system_message = EXPERT_SYSTEM_PROMPTS.get(
        expert_choice,
        "あなたは有能なアシスタントです。ユーザーの目的達成を最短で支援してください。"
    )

    # Lesson8スタイルのチェーン: PromptTemplate -> LLM -> StrOutputParser
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", "{user_input}")
    ])

    chain = prompt | llm | StrOutputParser()
    return chain.invoke({"user_input": input_text})


# ====== Streamlit UI ======
st.set_page_config(page_title="Lesson8風：専門家切替チャット", page_icon="🛠️", layout="centered")

st.title("🛠️ Lesson8風：専門家A/Bを切り替えて質問")
st.caption("Powered by LangChain + OpenAI")

with st.expander("📘 このWebアプリの概要と使い方", expanded=True):
    st.markdown(
        """
**できること**  
- 入力フォームにテキストを1つ入れて送信すると、LangChain経由でLLMに投げて回答を表示します。  
- ラジオボタンで「A（建設DXコンサル）」か「B（TikTokグロース）」を選ぶと、  
  それぞれの専門家としての **システムメッセージ** が挿し替わり、回答の切り口が変わります。

**使い方**  
1. 画面下のラジオボタンで専門家を選ぶ  
2. 入力ボックスに質問や相談内容を書く（例：「現場の安全点検を効率化したい」や「フォローが伸びない」）  
3. **送信** を押すと、回答が下に表示されます

**環境準備**  
- 事前に `OPENAI_API_KEY` を環境変数で設定してください  
- 実行コマンドの例：  
  - `pip install streamlit langchain langchain-openai`  
  - `streamlit run app.py`
        """
    )

with st.form("qa_form"):
    expert = st.radio(
        "専門家を選択：",
        options=list(EXPERT_SYSTEM_PROMPTS.keys()),
        horizontal=True
    )
    user_text = st.text_area(
        "入力テキスト",
        placeholder="例）工程遅延を最小化する現場運用の型を作りたい／3秒で掴むフック案を増やしたい など",
        height=140
    )
    submitted = st.form_submit_button("送信する 🚀")

if submitted:
    with st.spinner("専門家が考え中…"):
        try:
            answer = ask_expert(user_text, expert)
            st.success("回答が届きました")
            st.markdown("### ✍️ 回答")
            st.write(answer)
        except Exception as e:
            st.error(f"エラーが発生しました：{e}")

    with st.expander("🔧 現在の専門家プロンプト（Systemメッセージ）を確認する"):
        st.code(EXPERT_SYSTEM_PROMPTS.get(expert, ""), language="markdown")
