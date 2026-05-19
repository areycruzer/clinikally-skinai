"""Custom tool implementations for the Clinikally SkinAI agent pipeline.

This module defines the domain-specific tools that the SkinAI decision-tree
agent can invoke at runtime.  Each tool subclasses :class:`skinai.Tool` and
follows a consistent pattern:

1. **``__init__``** – registers the tool's name, natural-language description
   (consumed by the routing LLM), typed input schema, and a user-facing
   status message.
2. **``__call__``** – async generator that executes the tool logic, yielding
   one or more :class:`~skinai.objects.Response`, :class:`~skinai.objects.Retrieval`,
   or :class:`~skinai.objects.Error` objects back to the frontend via the
   WebSocket stream.

Tools defined here:

* :class:`ProductQueryTool` – hybrid vector search over the *SkincareProducts*
  Weaviate collection with optional price / skin-type filters.
* :class:`BlogRAGTool` – RAG over the *SkincareBlogs* Weaviate collection for
  educational and routine-related queries.
* :class:`GeneralKnowledgeTool` – pure LLM-based dermatological Q&A with no
  database dependency.
* :class:`SkinAnalysisTool` – multimodal vision analysis of user-uploaded skin
  photos using Gemini / OpenRouter models.
"""

from skinai import Tool
from skinai.objects import Response


class ProductQueryTool(Tool):
    """Hybrid vector-search tool for the *SkincareProducts* Weaviate collection.

    Builds a compound search query from the user's free-text description plus
    optional structured filters (price cap, skin type, category, ingredients,
    and concerns).  Results are formatted into rich product cards by the LLM
    and accompanied by a ``Retrieval`` payload for the frontend's card renderer.

    If the Weaviate database is unreachable, the tool transparently falls back
    to a general-knowledge LLM response so the user is never left without an
    answer.

    Attributes:
        name: ``"ProductQueryTool"``
        end: ``True`` – this tool produces a terminal response.
    """

    def __init__(self, **kwargs):
        super().__init__(
            name="ProductQueryTool",
            description="""
            Queries the SkincareProducts database to recommend products based on criteria.
            Use this tool for product queries and recommendation requests.
            """.strip(),
            inputs={
                "search_query": {
                    "type": str,
                    "description": "Description of what product or concern you are looking for.",
                    "required": True,
                },
                "price_limit": {
                    "type": float,
                    "description": "The maximum budget or price limit in ₹.",
                    "required": False,
                },
                "skin_type": {
                    "type": str,
                    "description": "The target skin type (e.g., 'oily', 'dry', 'sensitive', 'combination').",
                    "required": False,
                },
                "category": {
                    "type": str,
                    "description": "The product category (e.g., 'moisturiser', 'serum', 'facewash', 'sunscreen').",
                    "required": False,
                },
                "ingredients": {
                    "type": list[str],
                    "description": "Target active ingredients (e.g., ['niacinamide', 'salicylic acid']).",
                    "required": False,
                },
                "concerns": {
                    "type": list[str],
                    "description": "Skin concerns to address (e.g., ['acne', 'dark spots']).",
                    "required": False,
                },
            },
            status="Querying skincare products...",
            end=True,
        )

    async def __call__(
        self, tree_data, inputs, base_lm, complex_lm, client_manager, **kwargs
    ):
        """Execute a hybrid product search and generate recommendation cards.

        Args:
            tree_data: Decision-tree state including conversation history.
            inputs: Tool inputs parsed by the routing LLM.  Expected keys:
                ``search_query`` (str, required), ``price_limit`` (float),
                ``skin_type`` (str), ``category`` (str),
                ``ingredients`` (list[str]), ``concerns`` (list[str]).
            base_lm: Lightweight dspy LM (unused by this tool).
            complex_lm: High-capability dspy LM used for response synthesis.
            client_manager: Weaviate async client manager.

        Yields:
            Retrieval: Product objects with frontend-compatible field mapping.
            Response: LLM-generated Markdown product cards.
            Error: On synthesis failure.
        """
        import dspy
        from weaviate.classes.query import Filter
        from skinai.objects import Response, Error, Retrieval
        
        search_query = inputs.get("search_query", "")
        price_limit = inputs.get("price_limit")
        skin_type = inputs.get("skin_type")
        category = inputs.get("category")
        ingredients = inputs.get("ingredients")
        concerns = inputs.get("concerns")

        # Build dynamic search parameters
        query_parts = [search_query]
        if skin_type:
            query_parts.append(f"{skin_type} skin")
        if category:
            query_parts.append(category)
        if ingredients:
            if isinstance(ingredients, list):
                query_parts.extend(ingredients)
            else:
                query_parts.append(str(ingredients))
        if concerns:
            if isinstance(concerns, list):
                query_parts.extend(concerns)
            else:
                query_parts.append(str(concerns))
                
        full_search_query = " ".join(query_parts)
        
        # Build filters
        filters = []
        if price_limit is not None and price_limit > 0:
            filters.append(Filter.by_property("price").less_or_equal(price_limit))
            
        combined_filter = Filter.all_of(filters) if filters else None
        
        import asyncio
        products_data = []
        try:
            async with client_manager.connect_to_async_client() as client:
                coll = client.collections.get("SkincareProducts")
                res = await asyncio.wait_for(
                    coll.query.hybrid(
                        query=full_search_query,
                        filters=combined_filter,
                        limit=6
                    ),
                    timeout=10.0
                )
                for obj in res.objects:
                    products_data.append({
                        "title": obj.properties.get("title"),
                        "price": obj.properties.get("price"),
                        "category": obj.properties.get("category"),
                        "type": obj.properties.get("type"),
                        "tags": obj.properties.get("tags"),
                        "description": obj.properties.get("description"),
                        "url": obj.properties.get("url"),
                        "image_src": obj.properties.get("image_src"),
                        "vendor": obj.properties.get("vendor")
                    })
        except Exception as e:
            class GeneralKnowledgeFallback(dspy.Signature):
                """
                You are Clinikally's Expert Clinical Skincare Assistant.
                The product database is currently unreachable, so answer the user's query using general dermatological knowledge instead.
                """
                conversation_history: str = dspy.InputField(description="Previous conversation turns.")
                user_query: str = dspy.InputField(description="The user's query.")
                response: str = dspy.OutputField(description="Expert explanation in Markdown.")
            
            history_list = []
            if hasattr(tree_data, "conversation_history") and tree_data.conversation_history:
                for msg in tree_data.conversation_history[:-1]:
                    history_list.append(f"{msg['role']}: {msg['content']}")
            history_str = "\n".join(history_list) if history_list else "None."
            
            predictor = dspy.Predict(GeneralKnowledgeFallback)
            prediction = await predictor.aforward(
                conversation_history=history_str,
                user_query=full_search_query,
                lm=complex_lm
            )
            fallback_msg = "I'm having trouble accessing the product database right now, but here's what I know from general skincare knowledge...\n\n"
            yield Response(fallback_msg + prediction.response)
            return

        if not products_data:
            yield Response("I couldn't find any products matching your specific criteria in the Clinikally database. Please adjust your filters (like the price limit or skin type) and try again!")
            return
            
        class ProductQueryPrompt(dspy.Signature):
            """
            You are Clinikally's Expert Skincare Assistant.
            Your task is to take the retrieved skincare products, the user's request, and the previous conversation history, and generate a beautifully formatted response with product cards.
            
            Requirements for each product card:
            - Product Name (formatted bold, e.g., **Product Name**)
            - Price in ₹ (always use ₹ for Indian prices, e.g., ₹999)
            - Skin Suitability (based on tags/description, e.g. oily, dry, sensitive)
            - Key Benefits (bullet points summarizing description and active ingredients)
            - Source Link (the 'url' property of the product if available, format as a clickable link or button like [View Product](url))
            
            Format the output clearly and elegantly in Markdown. Be professional and encouraging. Use only the products in the context.
            """
            conversation_history: str = dspy.InputField(description="Previous conversation turns to maintain context.")
            user_query: str = dspy.InputField(description="The user's skincare product query.")
            products: list[dict] = dspy.InputField(description="List of retrieved skincare products with details.")
            response: str = dspy.OutputField(description="Curated list of product cards and recommendations in Markdown.")

        try:
            history_list = []
            if hasattr(tree_data, "conversation_history") and tree_data.conversation_history:
                for msg in tree_data.conversation_history[:-1]:
                    role = "User" if msg["role"] == "user" else "Assistant"
                    history_list.append(f"{role}: {msg['content']}")
            history_str = "\n".join(history_list) if history_list else "No previous conversation history."

            predictor = dspy.Predict(ProductQueryPrompt)
            prediction = await predictor.aforward(
                conversation_history=history_str,
                user_query=full_search_query,
                products=products_data,
                lm=complex_lm
            )
            
            # Map products to match what SkinAI frontend expects for rich product card rendering
            mapping = {
                "name": "title",
                "description": "description",
                "price": "price",
                "category": "category",
                "subcategory": "type",
                "tags": "tags",
                "url": "url",
                "image": "image_src",
                "brand": "vendor",
            }
            
            yield Retrieval(
                objects=products_data,
                metadata={
                    "collection_name": "SkincareProducts",
                    "query_text": full_search_query,
                    "return_type": "product",
                },
                payload_type="product",
                name="SkincareProducts",
                mapping=mapping,
            )
            
            yield Response(prediction.response)
        except Exception as e:
            yield Error(f"Error generating recommendation response: {str(e)}")


class BlogRAGTool(Tool):
    """RAG tool for the *SkincareBlogs* Weaviate collection.

    Performs hybrid search over Clinikally's curated blog corpus and
    synthesises an evidence-backed answer with cited source links.  Used for
    educational queries about routines, ingredients, seasonal skincare, etc.

    Falls back to general dermatological knowledge when the blog database is
    unreachable.

    Attributes:
        name: ``"BlogRAGTool"``
        end: ``True`` – this tool produces a terminal response.
    """

    def __init__(self, **kwargs):
        super().__init__(
            name="BlogRAGTool",
            description="""
            Queries the SkincareBlogs database to answer educational, routine, or ingredient questions.
            Use this tool for routine advice, ingredient explanations, and general summer/winter skincare queries.
            """.strip(),
            inputs={
                "search_query": {
                    "type": str,
                    "description": "Semantic search query to find relevant blog information.",
                    "required": True,
                },
                "tags": {
                    "type": list[str],
                    "description": "Relevant topic tags (e.g., ['Acne', 'Dry Skin']).",
                    "required": False,
                },
            },
            status="Searching skincare blogs...",
            end=True,
        )

    async def __call__(
        self, tree_data, inputs, base_lm, complex_lm, client_manager, **kwargs
    ):
        """Search skincare blogs and synthesise an educational answer.

        Args:
            tree_data: Decision-tree state including conversation history.
            inputs: Tool inputs.  Expected keys: ``search_query`` (str,
                required), ``tags`` (list[str], optional).
            base_lm: Lightweight dspy LM (unused by this tool).
            complex_lm: High-capability dspy LM used for answer synthesis.
            client_manager: Weaviate async client manager.

        Yields:
            Response: Synthesised Markdown answer with blog citations.
            Error: On synthesis failure.
        """
        import dspy
        from skinai.objects import Response, Error
        
        search_query = inputs.get("search_query", "")
        
        import asyncio
        blogs_data = []
        try:
            async with client_manager.connect_to_async_client() as client:
                coll = client.collections.get("SkincareBlogs")
                res = await asyncio.wait_for(
                    coll.query.hybrid(
                        query=search_query,
                        limit=5
                    ),
                    timeout=10.0
                )
                for obj in res.objects:
                    blogs_data.append({
                        "title": obj.properties.get("title"),
                        "summary": obj.properties.get("summary"),
                        "link": obj.properties.get("link"),
                        "content": obj.properties.get("content"),
                        "tags": obj.properties.get("tags")
                    })
        except Exception as e:
            class GeneralKnowledgeFallback(dspy.Signature):
                """
                You are Clinikally's Expert Clinical Skincare Assistant.
                The blog database is currently unreachable, so answer the user's query using general dermatological knowledge instead.
                """
                conversation_history: str = dspy.InputField(description="Previous conversation turns.")
                user_query: str = dspy.InputField(description="The user's query.")
                response: str = dspy.OutputField(description="Expert explanation in Markdown.")
            
            history_list = []
            if hasattr(tree_data, "conversation_history") and tree_data.conversation_history:
                for msg in tree_data.conversation_history[:-1]:
                    history_list.append(f"{msg['role']}: {msg['content']}")
            history_str = "\n".join(history_list) if history_list else "None."
            
            predictor = dspy.Predict(GeneralKnowledgeFallback)
            prediction = await predictor.aforward(
                conversation_history=history_str,
                user_query=search_query,
                lm=complex_lm
            )
            fallback_msg = "I'm having trouble accessing the clinical guides right now, but here's what I know from general skincare knowledge...\n\n"
            yield Response(fallback_msg + prediction.response)
            return

        if not blogs_data:
            yield Response("I couldn't find any specific Clinikally blog articles addressing this skincare topic. Let me share some general clinical knowledge instead!")
            return

        class BlogRAGPrompt(dspy.Signature):
            """
            You are Clinikally's Expert Skincare Assistant.
            Your task is to take the retrieved skincare blogs content, the user's query, and the previous conversation history, and synthesize a comprehensive, helpful, and highly accurate answer.
            
            Requirements:
            - Be highly professional, accurate, and scientific yet accessible.
            - Structure your response cleanly with headings and bullet points.
            - Always cite your sources! Include specific blog titles and direct links (from the 'link' property of the retrieved blogs, e.g. [Read the full article](link)) where the information was sourced.
            - If the retrieved blog content doesn't contain enough information to answer, state what you found and advise consulting a dermatologist, but do not hallucinate.
            """
            conversation_history: str = dspy.InputField(description="Previous conversation turns to maintain context.")
            user_query: str = dspy.InputField(description="The user's skincare query.")
            blogs_content: list[dict] = dspy.InputField(description="Retrieved skincare blog articles with content, summary, and link.")
            response: str = dspy.OutputField(description="Synthesized educational explanation with cited links in Markdown.")

        try:
            history_list = []
            if hasattr(tree_data, "conversation_history") and tree_data.conversation_history:
                for msg in tree_data.conversation_history[:-1]:
                    role = "User" if msg["role"] == "user" else "Assistant"
                    history_list.append(f"{role}: {msg['content']}")
            history_str = "\n".join(history_list) if history_list else "No previous conversation history."

            predictor = dspy.Predict(BlogRAGPrompt)
            prediction = await predictor.aforward(
                conversation_history=history_str,
                user_query=search_query,
                blogs_content=blogs_data,
                lm=complex_lm
            )
            yield Response(prediction.response)
        except Exception as e:
            yield Error(f"Error generating blog response: {str(e)}")


class GeneralKnowledgeTool(Tool):
    """LLM-only tool for general dermatological and skincare Q&A.

    Handles clinical explanations, medical causes (e.g. hormonal acne),
    routine guidance, and lifestyle advice without requiring any database
    lookup.  The response is generated entirely by the complex LM.

    Attributes:
        name: ``"GeneralKnowledgeTool"``
        end: ``True`` – this tool produces a terminal response.
    """

    def __init__(self, **kwargs):
        super().__init__(
            name="GeneralKnowledgeTool",
            description="""
            Answers general skincare questions, clinical explanations, medical causes (like hormonal acne), or how-to's using expert clinical knowledge.
            """.strip(),
            inputs={
                "search_query": {
                    "type": str,
                    "description": "The user's general skincare question or concern.",
                    "required": True,
                },
            },
            status="Applying clinical skincare knowledge...",
            end=True,
        )

    async def __call__(
        self, tree_data, inputs, base_lm, complex_lm, client_manager, **kwargs
    ):
        """Generate an expert skincare answer using LLM knowledge.

        Args:
            tree_data: Decision-tree state including conversation history.
            inputs: Tool inputs.  Expected key: ``search_query`` (str).
            base_lm: Lightweight dspy LM (unused by this tool).
            complex_lm: High-capability dspy LM used for response generation.
            client_manager: Weaviate async client manager (unused).

        Yields:
            Response: Expert Markdown explanation.
            Error: On generation failure.
        """
        import dspy
        from skinai.objects import Response, Error
        
        search_query = inputs.get("search_query", "")

        class GeneralKnowledgePrompt(dspy.Signature):
            """
            You are Clinikally's Expert Clinical Skincare Assistant.
            Your task is to answer general or medical skincare questions (e.g. hormonal acne, dark circles, skin routines) using expert dermatological knowledge and maintaining the previous conversation history.
            
            Requirements:
            - Provide helpful, accurate, structured, and clinically sound explanations.
            - Frame advice professionally, explaining underlying causes (e.g., hormonal fluctuations, genetics, diet) and offering actionable steps (routines, active ingredients to look for, lifestyle changes).
            - Maintain a caring and expert tone.
            - Always recommend consulting a certified dermatologist for persistent or severe medical concerns.
            - Use ₹ for any price references, and keep formatting beautiful in Markdown.
            """
            conversation_history: str = dspy.InputField(description="Previous conversation turns to maintain context.")
            user_query: str = dspy.InputField(description="The user's general skincare question.")
            response: str = dspy.OutputField(description="Expert, well-structured dermatological explanation in Markdown.")

        try:
            history_list = []
            if hasattr(tree_data, "conversation_history") and tree_data.conversation_history:
                for msg in tree_data.conversation_history[:-1]:
                    role = "User" if msg["role"] == "user" else "Assistant"
                    history_list.append(f"{role}: {msg['content']}")
            history_str = "\n".join(history_list) if history_list else "No previous conversation history."

            predictor = dspy.Predict(GeneralKnowledgePrompt)
            prediction = await predictor.aforward(
                conversation_history=history_str,
                user_query=search_query,
                lm=complex_lm
            )
            yield Response(prediction.response)
        except Exception as e:
            yield Error(f"Error generating general knowledge response: {str(e)}")


class SkinAnalysisTool(Tool):
    """Multimodal vision tool for analysing user-uploaded skin photos.

    Sends the base-64-encoded skin image to a vision-capable LLM (Gemini or
    OpenRouter fallback chain) together with a clinical dermatology prompt.
    Returns a structured diagnostic report covering skin type, visible
    concerns, recommended routines, and professional advice.

    The tool implements a multi-provider retry chain so that transient API
    failures are handled gracefully without falling back to a text-only tool
    that cannot interpret images.

    Attributes:
        name: ``"SkinAnalysisTool"``
        end: ``True`` – this tool produces a terminal response.
    """

    def __init__(self, **kwargs):
        super().__init__(
            name="SkinAnalysisTool",
            description="""
            Analyzes an uploaded skin photo to diagnose skin concerns and recommend customized clinical skincare solutions.
            Use this tool when the user uploads an image/photo of their skin for analysis, diagnosis, or routine recommendations.
            """.strip(),
            inputs={
                "search_query": {
                    "type": str,
                    "description": "Optional search query or description of concerns provided by the user.",
                    "required": False,
                },
            },
            status="Analyzing uploaded skin photo...",
            end=True,
        )

    async def __call__(
        self, tree_data, inputs, base_lm, complex_lm, client_manager, **kwargs
    ):
        """Analyse an uploaded skin photo and produce a clinical report.

        The method reads the base-64 image from ``tree_data.last_uploaded_image``,
        constructs a multimodal prompt, and attempts inference through a
        prioritised list of vision models (Gemini → OpenRouter free tier).
        On success the image reference is cleared to prevent re-analysis on
        subsequent turns.

        Args:
            tree_data: Decision-tree state; must carry
                ``last_uploaded_image`` (base-64 data URI).
            inputs: Tool inputs.  Optional key: ``search_query`` (str) for
                additional user context.
            base_lm: Lightweight dspy LM (unused by this tool).
            complex_lm: High-capability dspy LM (unused – vision calls go
                through ``litellm`` directly).
            client_manager: Weaviate async client manager (unused).

        Yields:
            Response: Markdown clinical diagnostic report, or a user-friendly
                error message if all vision providers fail.
        """
        import os
        import asyncio
        import litellm
        from skinai.objects import Response, Error
        
        search_query = inputs.get("search_query", "")
        
        # Check if last_uploaded_image is present in tree_data
        image_data = getattr(tree_data, "last_uploaded_image", None)
        if not image_data:
            yield Response(
                "It looks like you want me to analyze your skin, but no photo has been uploaded yet! 📸\n\n"
                "Please click the **📸 camera icon** next to the chat bar or **drag and drop a skin photo** directly "
                "into the input box, then send your query. I'll be happy to analyze it for you! ✨"
            )
            return

        # Prepare multimodal message
        prompt = (
            "You are Clinikally's Senior Consultant Dermatologist and Skincare AI Specialist.\n"
            "Analyze the attached skin photo carefully and provide a premium clinical skincare diagnostic report.\n\n"
            "Your response must include:\n"
            "1. **Visual Skincare Diagnostic**: Identify the likely skin type (Oily, Dry, Sensitive, Combination) "
            "and diagnose visible concerns (e.g., acne severity, texture/pores, pigmentation/dark spots, redness/sensitivity, fine lines/wrinkles).\n"
            "2. **Clinical Routine Recommendation**: A step-by-step skincare routine (Morning/Evening) featuring "
            "clinical active ingredients (e.g., Niacinamide, Salicylic Acid, Retinol, Vitamin C) tailored to address the concerns shown in the photo.\n"
            "3. **Professional Dermatological Advice**: Helpful tips, lifestyle guidance, and a recommendation on when to consult a certified dermatologist.\n\n"
            "Requirements:\n"
            "- Format your response in beautiful, professional Markdown with neat headers and clear bullet points.\n"
            "- Be precise, warm, highly professional, and encouraging.\n"
            "- Keep currency in ₹ for any price or product mentions (if applicable).\n"
        )
        if search_query:
            prompt += f"- Address the user's specific text question/concern: \"{search_query}\"\n"

        try:
            gemini_key = os.getenv("GEMINI_API_KEY")
            openrouter_key = os.getenv("OPENROUTER_API_KEY")
            
            import logging
            logger = logging.getLogger(__name__)
            logger.info(f"[SkinAnalysis] GEMINI_API_KEY loaded: {gemini_key[:10]}...{gemini_key[-4:] if gemini_key else 'NONE'}")
            logger.info(f"[SkinAnalysis] OPENROUTER_API_KEY loaded: {openrouter_key[:10]}...{openrouter_key[-4:] if openrouter_key else 'NONE'}")
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_data}}
                    ]
                }
            ]
            
            # Retry chain: Gemini direct first, then OpenRouter free vision models as fallback
            attempts = [
                {"model": "gemini/gemini-2.5-flash", "api_key": gemini_key},
                {"model": "gemini/gemini-2.0-flash", "api_key": gemini_key},
                {"model": "openrouter/google/gemma-4-31b-it:free", "api_key": openrouter_key},
                {"model": "openrouter/nvidia/nemotron-nano-12b-v2-vl:free", "api_key": openrouter_key},
                {"model": "openrouter/nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free", "api_key": openrouter_key},
            ]
            last_error = None
            
            for i, spec in enumerate(attempts):
                try:
                    logger.info(f"[SkinAnalysis] Attempt {i+1}/{len(attempts)}: trying {spec['model']}")
                    response = await asyncio.to_thread(
                        litellm.completion,
                        model=spec["model"],
                        messages=messages,
                        api_key=spec["api_key"],
                    )
                    analysis_result = response.choices[0].message.content
                    logger.info(f"[SkinAnalysis] SUCCESS with {spec['model']}")
                    
                    # Clear the uploaded image state to prevent repeating on next turns
                    if hasattr(tree_data, "last_uploaded_image"):
                        tree_data.last_uploaded_image = None
                        
                    yield Response(analysis_result)
                    return
                except Exception as retry_err:
                    logger.error(f"[SkinAnalysis] FAILED {spec['model']}: {type(retry_err).__name__}: {str(retry_err)[:200]}")
                    last_error = retry_err
                    if i < len(attempts) - 1:
                        await asyncio.sleep(2)  # Brief pause between attempts
            
            # All retries exhausted — yield a Response (NOT Error) so the framework
            # does NOT fall back to GeneralKnowledgeTool which cannot see images.
            yield Response(
                "⚠️ **Temporary Service Issue**\n\n"
                "I was able to receive your skin photo, but the vision analysis model is currently "
                "experiencing high demand. Please try again in a few moments!\n\n"
                f"*Technical detail: {str(last_error)[:200]}*"
            )
        except Exception as e:
            # Catch-all: still yield Response to prevent blind fallback
            yield Response(
                "⚠️ **Skin Analysis Error**\n\n"
                f"An unexpected error occurred while analyzing your photo: {str(e)[:300]}\n\n"
                "Please try uploading again or check back shortly."
            )
