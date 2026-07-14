from app.web.skill_rag import SkillDocument, retrieve_relevant_skills


def test_skill_rag_routes_task_to_relevant_skill():
    documents = [
        SkillDocument(
            id="gmail",
            name="Gmail 邮箱处理",
            summary="搜索、阅读、回复和发送 Gmail 邮件。",
            content="适合邮箱检索、邮件回复、草稿、附件下载和 Gmail MCP 工具调用。",
        ),
        SkillDocument(
            id="office",
            name="文档生成",
            summary="生成 Word、Excel 和 PDF 文档。",
            content="适合将论文、网页和表格资料整理成 docx、xlsx、pdf。",
        ),
        SkillDocument(
            id="browser",
            name="浏览器自动化",
            summary="使用 Playwright 浏览网页、点击、填写表单和截图。",
            content="适合网页搜索、打开页面、读取 DOM、输入内容、提交表单和截图验证。",
        ),
    ]

    matches = retrieve_relevant_skills(
        "帮我查看 Gmail 邮箱里导师最近发来的邮件并起草回复",
        documents,
        top_k=2,
        min_score=0.01,
    )

    assert matches
    assert matches[0].id == "gmail"
    assert "gmail" in matches[0].matched_terms


def test_skill_rag_excludes_manually_selected_skill():
    documents = [
        SkillDocument(
            id="gmail",
            name="Gmail 邮箱处理",
            summary="搜索、阅读、回复和发送 Gmail 邮件。",
            content="适合邮箱检索、邮件回复、草稿、附件下载和 Gmail MCP 工具调用。",
        ),
        SkillDocument(
            id="browser",
            name="浏览器自动化",
            summary="使用 Playwright 浏览网页、点击、填写表单和截图。",
            content="适合网页搜索、打开页面、读取 DOM、输入内容、提交表单和截图验证。",
        ),
    ]

    matches = retrieve_relevant_skills(
        "帮我发送 Gmail 邮件",
        documents,
        exclude_ids={"gmail"},
        top_k=2,
        min_score=0.01,
    )

    assert all(match.id != "gmail" for match in matches)


def test_skill_rag_recognizes_luogu_url_and_colloquial_submit_request():
    documents = [
        SkillDocument(
            id="luogu-submit",
            name="Luogu Problem Submission",
            summary="洛谷做题、本地测试、提交代码并检查评测结果。",
            content=(
                "Use for luogu.com.cn/problem URLs and requests such as "
                "自己做这道题、做出来交上、交一下。"
            ),
        )
    ]

    matches = retrieve_relevant_skills(
        "https://www.luogu.com.cn/problem/P17036，自己把这道题做出来交上。",
        documents,
        top_k=1,
        min_score=0.01,
    )

    assert matches
    assert matches[0].id == "luogu-submit"
