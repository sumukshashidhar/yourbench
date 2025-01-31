# Sacra Huggingface Analysis

Sacra estimates that Hugging Face hit $70M annual recurring revenue (ARR) at the end of 2023, up 367% from 2022, primarily off the growth of lucrative consulting contracts with many of the biggest companies in AI—Nvidia, Amazon, and Microsoft. 

Like Github, Hugging Face makes money from paid individual ($9/month) and team plans ($20/month), but the majority of their revenue comes from spinning up closed, managed versions of their product for the enterprise.

Product
None
Hugging Face (2017) found product-market fit as the Github of ML, allowing any developer with a bit of Python knowledge to quickly get started, with 20K+ open datasets, one-button LoRa fine-tuning bundled into their Transformers library (121K stars on Github), and 120K+ pre-trained open-source models.

Researchers open-source their models and upload them to Hugging Face because while it takes millions of dollars in compute to build a consumer product like ChatGPT, distributing your model via Hugging Face gets it in front of every ML developer for free—and leaves open the potential to charge for an API version of it down the line.

Hugging Face's main product is a library of pre-trained models and tools for NLP tasks. Think of these models as trained experts in understanding and processing language. For example, Hugging Face has models that can classify text into different categories (like sentiment analysis), extract information from text, translate text, generate text, and more.

These models are trained on a large amount of text data, which allows them to learn the patterns and structures of human language. Once the models are trained, they can be fine-tuned on specific NLP tasks with a smaller amount of data in what’s known as transfer learning.

Hugging Face also operates a cloud platform that provides a range of NLP and AI services, including model hosting, inference, and optimization, so that users can deploy, manage, and scale their NLP models in a cloud environment.

Business Model
None
Hugging Face was founded in 2016 as a chatbot for teens, but they found product-market fit in 2018 after they open-sourced the Transformers library they built to run it, consisting of a set of 30 pre-trained models for text classification and summarization with APIs so any developer could quickly train those models on their own data.

On top of that core library, Hugging Face has built a Github-like cloud platform for collaborating on and sharing those models, with 500,000 user-contributed open-source models available for developers to use for training.

Today, Hugging Face's primary sources of revenue include:

Team subscriptions
Hugging Face’s basic product consists of premium access to their database of models, with two basic tiers between “pro” and “enterprise”.

Pro costs $9 per month and offers private dataset viewing, inference capabilities, and early access to new features.

Enterprise is tailored to bigger teams and is priced per-user at $20 per month, and it offers single sign-on (SSO), regional data storage, audit logs, access control, the ability to run inference on your own infrastructure, and priority support.

Cloud services
Hugging Face operates a cloud platform that provides a range of NLP and AI services, including model hosting, inference, and optimization. The platform allows users to easily deploy, manage, and scale their NLP models in a cloud environment. Hugging Face charges its cloud users based on usage, with fees for model hosting, inference, and optimization.

Enterprise
Hugging Face offers a range of enterprise solutions that leverage its NLP and AI technology. These solutions include custom model training, deployment, and integration services, as well as access to premium features and support. Hugging Face charges its enterprise customers for these services on a contract basis.

So far, Hugging Face has partnered with companies like Nvidia to link Hugging Face users to compute via Nvidia’s DGX platform, Amazon to bring Hugging Face to AWS, and Microsoft to spin up Hugging Face Endpoints on Azure.

Competition
Hugging Face's model-agnostic, collaborative workspace provides them with unique positioning in the AI space. 

By allowing teams to e.g. leverage high-quality closed-source models to train their own low-cost, open-source models, Hugging Face has exposure both to the upside of proprietary models getting more advanced and open-source models getting cheaper and faster to train.

While the companies behind individual models may rise and fall, Hugging Face both perpetuates and benefits from increasing non-monogamy among ML developers, who prefer mixing and matching models to find the best solution for their particular use case.

OpenAI
OpenAI hit $2B in annual recurring revenue at the end of 2023, up about 900% from $200M at the end of 2022.

OpenAI was founded in December 2015 as a non-profit dedicated to developing “safe” artificial intelligence. Its founding team included Sam Altman, Elon Musk, Greg Brockman, Jessica Livingston, and others.

OpenAI’s first products were released in 2016—Gym, their reinforcement learning research platform, and Universe, their platform for measuring the intelligence of artificial agents engaged in playing videogames and performing other tasks.

OpenAI’s flagship consumer product today is ChatGPT, which millions of people use everyday for tasks like code generation, research, Q&A, therapy, medical diagnoses, and creative writing.

OpenAI has about two dozen different products across AI-based text, images, and audio generation, including its GPT-3 and GPT-4 APIs, Whisper, DALL-E, and ChatGPT.

Google
Earlier this year, Google merged its DeepMind and Google Brain AI divisions in order to develop a multi-modal AI model to go after OpenAI and compete directly with GPT-4 and ChatGPT. The model is currently expected to be released toward the end of 2023.

Gemini is expected to have the capacity to ingest and output both images and text, giving it the ability to generate more complex end-products than a text-alone interface like ChatGPT.

One advantage of Google’s Gemini is that it can be trained on a massive dataset of consumer data from Google’s various products like Gmail, Google Sheets, and Google Calendar—data that OpenAI cannot access because it is not in the public domain.

Another massive advantage enjoyed by Google here will be their vast access to the most scarce resource in AI development—compute.

No company has Google’s access to compute, and their mastery of this resource means that according to estimates, they will be able to grow their pre-training FLOPs (floating point operations per second) to 5x that of GPT-4 by the end of 2023 and 20x by the end of 2024.

Meta
Meta has been a top player in the world of AI for years despite not having the outward reputation of a Google or OpenAI—software developed at Meta like Pytorch, Cicero, Segment Anything and RecD have become standard-issue in the field.

When Meta’s foundation model LLaMA leaked to the public in March, it immediately caused a stir in the AI development community—where previously models trained on so many tokens (1.4T in the case of LLaMa) had been the proprietary property of companies like OpenAI and Google, in this case, the model became “open source” for anyone to use and train themselves.

When it comes to advantages, Meta—similar to Google—has the benefit of compute resources that they can use both for developing their LLMs and for recruiting the best talent. Meta have the 2nd most H100 GPUs in the world, behind Google.

Anthropic
Anthropic is an AI research company started in 2021 by Dario Amodei (former VP of research at OpenAI), Daniela Amodei (former VP of Safety and Policy at OpenAI) and nine other former OpenAI employees, including the lead engineer on GPT-3, Tom Brown. Their early business customers include Notion, DuckDuckGo, and Quora.

Notion uses Anthropic to power Notion AI, which can summarize documents, edit existing writing, and generate first drafts of memos and blog posts.

DuckDuckGo uses Anthropic to provide “Instant Answers”—auto-generated answers to user queries.

Quora uses Anthropic for their Poe chatbot because it is more conversational and better at holding conversation than ChatGPT.

In March 2023, Anthropic launched its first product available to the public—the chatbot Claude, competitive with ChatGPT. Claude’s 100K token context window vs. the roughly 4K context window of ChatGPT makes it potentially useful for many use cases across the enterprise.

TAM Expansion
There are a variety of routes that Hugging Face can take both to increase its monetization of its existing market and grow that market further.

Enterprise pricing
Similar to early Github, Hugging Face appears to be under-monetized. 

Where Github gave away unlimited public repos for free and charged businesses just $25/month to create private repos, Hugging Face lets users host unlimited models, datasets, and spaces, public or private, for free. 

The vast majority of their ~$70M in annualized revenue today comes from the managed version of their product they’re selling into companies like Amazon, Nvidia, and Microsoft.

The upshot—Hugging Face is betting on there being a much bigger monetization opportunity in the future as a result of being the central collaborative tool for devs building with AI.

Vertical solutions
Most of what Hugging Face offers today is still aimed at software developers with a working knowledge of Python, though they no longer need to have any real familiarity with machine learning per se to get started.

But Hugging Face can expand their TAM significantly by offering no-code and low-code workflows for spinning up specific kinds of AI-based products for various verticals.

Early in February 2024, Hugging Face launched one of these kinds of products with HuggingChat—a web app that lets users create personalized AI chatbots for free, leveraging the thousands of text-based language models that have been uploaded to their platform.

Building and launching more of these kinds of apps can allow Hugging Face to go after end-to-end seats within organizations, capturing product managers and others who might be building apps in Zapier and Airtable today.