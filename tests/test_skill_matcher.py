from app.web.skill_matcher import SkillDocument, match_skills


def skill_documents():
    return [
        SkillDocument(
            id="gmail",
            name="Gmail 邮箱处理",
            summary="搜索未读邮件并起草或发送回复。",
            content="处理收件箱、发件人、主题、附件和邮件线程。",
        ),
        SkillDocument(
            id="office",
            name="Word 与 PDF 文档生成",
            summary="生成正式 Word、DOCX 和 PDF 报告。",
            content="创建标题、段落、表格并检查最终排版。",
        ),
        SkillDocument(
            id="browser",
            name="浏览器自动化",
            summary="使用 Playwright 打开网页、点击并填写表单。",
            content="适合网站导航、页面读取、提交和截图验证。",
        ),
    ]


def test_tfidf_skill_matcher_routes_explicit_capability_terms():
    matches = match_skills(
        "打开 Gmail 收件箱，检查未读邮件并回复导师",
        skill_documents(),
        top_k=2,
        min_score=0.01,
    )

    assert matches
    assert matches[0].id == "gmail"
    assert matches[0].retrieval_method == "tfidf"
    assert matches[0].matched_terms


def test_tfidf_skill_matcher_excludes_manually_selected_skill():
    matches = match_skills(
        "生成 Word 报告",
        skill_documents(),
        exclude_ids={"office"},
        top_k=3,
        min_score=0.0,
    )

    assert all(match.id != "office" for match in matches)


def test_search_word_routes_to_web_research_skill_via_intent_aliases():
    documents = skill_documents() + [
        SkillDocument(
            id="web-research",
            name="联网检索与资料调研",
            summary="检索最新资料，阅读官方网页并核验事实。",
            content="先定位权威来源，再打开原始页面形成带来源的结论。",
        )
    ]

    matches = match_skills(
        "搜索某大学实验室2026届毕业去向",
        documents,
        top_k=3,
        min_score=0.08,
    )

    assert matches
    assert matches[0].id == "web-research"
    assert "检索" in matches[0].matched_terms


def test_mailbox_search_keeps_email_skill_ahead_of_web_research():
    documents = skill_documents() + [
        SkillDocument(
            id="web-research",
            name="联网检索与资料调研",
            summary="检索最新资料，阅读官方网页并核验事实。",
            content="先定位权威来源，再打开原始页面形成带来源的结论。",
        )
    ]

    matches = match_skills(
        "打开 Gmail 搜索导师邮件",
        documents,
        top_k=3,
        min_score=0.01,
    )

    assert matches[0].id == "gmail"
