# Human Preference Learning

\begin{abstract}
Large language models demonstrate impressive reasoning abilities but struggle to provide personalized content due to their lack of individual user preference information. Existing methods, such as in-context learning and parameter-efficient fine-tuning, fall short in capturing the complexity of human preferences, especially given the small, personal datasets individuals possess. In this paper, we propose a novel approach utilizing small parameter models as preference agents to generate natural language rules that guide a larger, pre-trained model, enabling efficient personalization. Our method involves a small, local "steering wheel" model that directs the outputs of a much larger foundation model, producing content tailored to an individual's preferences while leveraging the extensive knowledge and capabilities of the large model. Importantly, this personalization is achieved without the need to fine-tune the large model. Experimental results demonstrate that our technique significantly outperforms baseline personalization methods. By allowing foundation models to adapt to individual preferences in a data and compute-efficient manner, our approach paves the way for highly personalized language model applications.
\end{abstract}

\section{Introduction}

Large Language Models have revolutionized various domains with their impressive capabilities in reasoning, knowledge retrieval, and content generation. However, a crucial aspect where they often fall short is the ability to adapt their outputs to individual user preferences. While LLMs excel at generating content that caters to a broad audience, they struggle to produce outputs that resonate with the unique stylistic and contextual needs of individual users \cite{woźniak2024personalized}. This limitation stems from the fact that LLMs are typically trained on massive, generic datasets, promoting neutrality and hindering their capacity to learn and adapt to individual nuances \cite{doi:10.14318/hau6.1.002}.

Existing personalization techniques, such as in-context learning (ICL) \cite{NEURIPS2020_1457c0d6} and parameter-efficient fine-tuning (PEFT) \cite{hu2021lora, dettmers2023qlora}, have shown promise in adapting LLMs to specific tasks. However, these methods encounter significant challenges when applied to the domain of human preference learning. ICL, while effective in providing general guidance, struggles to capture the intricate and often contradictory nature of human preferences, especially when relying on limited in-context examples \cite{peng2023does}. Similarly, PEFT methods, while efficient in terms of compute and storage, face difficulties in generalizing to new preferences, particularly when users only possess small, personal datasets \cite{balne2024parameter}. This raises a fundamental question: \textbf{how can we efficiently and effectively align powerful LLMs to individual user preferences, especially when personalized data is scarce?}

To address this challenge, we propose a novel approach based on the concept of preference agents. These agents are small, locally trainable language models designed to learn and encode individual user preferences into concise natural language rules. These agents act like a small "steering wheel," guiding the output of a much larger, generic LLM towards the desired personalized style and content. This modular architecture decouples the task of preference learning from the generic LLM, allowing users to efficiently fine-tune a small agent on their personal data without the need for modifying the weights of the larger model.

Our approach represents a significant departure from conventional preference learning methods, offering a potential solution for unsupervised human preference learning. We evaluate our method across three diverse datasets encompassing human-generated content: emails, news articles and product reviews. Our results demonstrate that preference-guided LLMs significantly outperform both fine-tuning baselines and standard prompting techniques, based on automatic metrics, GPT-4o evaluation, and human judgments.

Specifically, our contributions include:

\begin{itemize}
\item A novel fine-tuning objective that leverages distilled preference information instead of traditional input-output pairs, promoting efficient learning of user preferences.
\item Empirical evidence that the use of preference agents leads to significant performance improvements – up to 80\% in some cases – over existing personalization methods, particularly when aligning LLMs to individual styles and preferences.
\item The release of three large, human intent annotated preference datasets to foster future research in personalization. \footnote{\href{https://huggingface.co/datasets/preference-agents/}{https://huggingface.co/preference-agents}}
\end{itemize}

\section{Method}


\begin{figure*}[t!]
    \centering
    \includegraphics[width=\textwidth]{resources/images/method/PrefAgents_nn.png}
     \caption{Preference Rule Finetuning vs Naive Finetuning and Large Model Zero-Shot (Figure WIP)}
    \label{fig:pref-agents-teaser}
\end{figure*}

In this section, we detail our approach for aligning language models to personalized user preferences using small preference agents. Our method involves two key components: generating natural language rules that capture user preferences and utilizing these rules to guide a larger, pre-trained language model. This modular architecture allows for efficient personalization without extensive retraining.

\subsection{Task Definition}


Given a task \( T \), we define the dataset \( \mathcal{D} \) as consisting of input-output pairs. Each input comprises a user intent \( \mathbf{u} \) and associated task metadata \( \mathbf{m} \), and the output is the ideal task completion, denoted as \( \mathbf{g} \), which we consider the ground truth. Thus, the dataset can be formally expressed as:

\[
\mathcal{D} = \{(\mathbf{x}, \mathbf{g}) \mid \mathbf{x} = (\mathbf{u}, \mathbf{m})\}
\]

\subsection{Constraints and Assumptions}

We seek to enable users to generate high quality, personalized responses as our goal, which are bounded by some constraints and assumptions:
\begin{itemize}
    \item \textbf{Constraint 1:} The size of the dataset $\mathcal{D}$ is not large enough to permit effective full parameter model fine-tuning. Given that individual users typically possess small, personal datasets, it is impractical to expect these datasets to be sufficient for extensive fine-tuning of a large language model.
    \item \textbf{Constraint 2:} The small model, denoted as $M_S$, must be lightweight enough to operate (w.r.t both training and inference) on lower power end-user devices. This requirement ensures that users can generate and apply their preferences without the need for high-performance computing resources. This allows for local inference, making the personalization process more accessible and convenient.
    \item \textbf{Constraint 3:} The large model, referred to as $M_L$, is either too large to run inference locally or is a closed-source API model. Consequently, it is not feasible, or cost effective to fine-tune or align $M_L$ by altering its model weights.
\end{itemize}

\subsection{Model Training}
\label{section:method/model-training}

Given the dataset $\mathcal{D}$, we first task $M_L$ with generating zero-shot responses to our training data. These initial responses are devoid of any user-specific preference information:

\begin{equation}
\mathbf{Y}_z = M_L(\mathbf{X})
\end{equation}

\noindent
where $\mathbf{Y}_z$ represents the set of zero-shot outputs for all inputs $\mathbf{X}$ in the training dataset.

Next, we leverage $M_L$'s capabilities to extract the delta between the zero-shot completions ($\mathbf{Y}_z$) and the ground truth outputs ($\mathbf{G}$). This delta represents the preference rules that need to be learned by the smaller model:

\begin{equation}
\mathbf{P} = M_L(\mathbf{Y}_z, \mathbf{G})
\end{equation}

Here, $\mathbf{P}$ represents the set of preference rules derived for each training example.  We hypothesize that $M_L$ can effectively identify these rules without prior knowledge of the specific user's preferences, just by observing the differences between the zero shot completion and the ground truth.

Finally, we train the smaller model, $M_S$, to learn to generate these preference rules.  The training data for $M_S$ consists of input-preference rule pairs:

\begin{equation}
M_S \xrightarrow{(\mathbf{X}, \mathbf{P})} M_A
\end{equation}

Through this training process, $M_S$ learns to map user intents and task metadata to natural language preference rules, effectively becoming a personalized preference agent ($M_A$).


\subsection{Model Alignment}

Once the preference agent $M_A$ is trained, we can use it to align the larger model's outputs to unseen user data. For a new input $\mathbf{x}$, we first generate preference rules using the trained agent:

\begin{equation}
\mathbf{p} = M_A(\mathbf{x})
\end{equation}

These rules, expressed in natural language, are then provided as additional context to the large language model $M_L$ alongside the original input:

\begin{equation}
y_a = M_L(\mathbf{x}, \mathbf{p})
\end{equation}

The output $y_a$ is considered to be preference-aligned as it is generated by $M_L$ while considering the user's preferences encoded in $\mathbf{p}$. This approach allows us to leverage the vast knowledge of $M_L$ while tailoring the output to individual preferences without directly modifying the large model's weights.

\subsection{Quantifying Alignment}

We utilize an evaluation function $\text{Eval}(y_a, y_z | \mathbf{x})$ on an unseen test set $\mathcal{T}$. For each example in $\mathcal{T}$, the function compares the preference-aligned output $y_a$ with the zero-shot output $y_z$ generated by $M_L$ without preference rules. The evaluation function assigns a score indicating the preference between $y_a$ and $y_z$, given the input $\mathbf{x}$. A positive score indicates a preference for the aligned output $y_a$, suggesting better alignment with the user's likely preference, while a negative score favors the zero-shot output. We aggregate these scores across all examples in $\mathcal{T}$ to obtain an overall alignment score:

\begin{equation}
\text{Score}(\mathcal{T}) = \sum_{i=1}^{|\mathcal{T}|} \text{Eval}(y_{a}^{(i)}, y_{z}^{(i)} | \mathbf{x}^{(i)})
\end{equation}

where $|\mathcal{T}|$ represents the number of examples in the test set and $y_{a}^{(i)}$ and $y_{z}^{(i)}$ represent the aligned and zero-shot outputs, respectively, for the $i$-th example.

A positive $\text{Score}(\mathcal{T})$ indicates that the preference agent successfully guides the LLM to generate outputs better aligned with user preferences compared to the baseline zero-shot outputs.

\section{Experimental Setup}

\subsection{Model Choice}

We employ Llama-3-8B-Instruct as our smaller, locally inferrable model ($M_S$) due to its strong capabilities and suitability for QLoRA fine-tuning on consumer hardware \cite{dettmers2023qlora}. For our larger reasoning model ($M_L$), we utilize Llama-3-70B-Instruct \cite{llama3modelcard}, Claude 3.5 Sonnet \cite{anthropic2024claude35}, and Gemini 1.5 Pro \cite{geminiteam2024gemini15unlockingmultimodal}. This diverse selection allows us to evaluate our approach with both a strong open-source model (Llama-3-70B-Instruct) and powerful closed-source API models (Claude 3.5, Gemini 1.5), to demonstrate generalizability.



\subsection{Datasets}

Our evaluation spans three datasets encompassing single and multi-user preference information:

\textbf{Enron Email Corpus.} For evaluating short-form writing personalization, we utilize the Enron email corpus \cite{10.1007/978-3-540-30115-8_22}, comprising emails from approximately 150 users, primarily senior management at Enron. We sample 15 users to analyze the reproducibility of individual writing styles. Each user's subset is split into an 80-20 train-test split.

\textbf{New Yorker.} To assess performance on long-form creative writing, we employ a subset of the All the News 2.0 dataset \cite{allthenews2}, specifically articles from The New Yorker magazine. This subset, containing approximately 3,500 articles, provides a rich source of author preference information. We investigate whether our preference agents can reproduce the unique style of The New Yorker using natural language rules. We split this dataset into a 50-50 train-test split.

\textbf{Amazon Review Subset (LAMP 3U)}. To further assess the generalizability of our framework beyond long-form content, we incorporate the LAMP 3U dataset \cite{salemi2024lamplargelanguagemodels}, which consists of Amazon product reviews grouped by user. We sampled 15 random users from LAMP 3U and then generated review intents and rules for each user following the same process employed for the Enron dataset.

Refer to Appendix \ref{appendix:dataset-enron} for details regarding dataset preparation and sampling.

\begin{table}[ht!]
\centering
\scalebox{0.8}{
\begin{tabular}{@{}l>{\raggedleft\arraybackslash}p{2.5cm}@{}}
\toprule
\textbf{Metric} & \textbf{Value} \\
\midrule
\multicolumn{2}{@{}c}{\textit{\textbf{Enron-42K (Short Form)}}} \\
\cmidrule(l){1-2}
Number of Data Points & 40,240 \\
Number of Unique Senders & 191 \\
Avg. Token Count (Email Content) & 58.83 \\
Avg. Token Count (Previous Context) & 261.48 \\
\midrule
\multicolumn{2}{@{}c}{\textit{\textbf{LAMP 3U (Medium Form)}}} \\
\cmidrule(l){1-2}
Number of Data Points & 22,500 \\
Number of Unique Users & 15 \\
Avg. Token Count (Review) & 144.32 \\
\midrule
\multicolumn{2}{@{}c}{\textit{\textbf{New Yorker (Long Form)}}} \\
\cmidrule(l){1-2}
Number of Data Points & 1,525 \\
Number of Unique Article Authors & 401 \\
Avg. Token Count (Article) & 846.60 \\
\bottomrule
\end{tabular}
}
\caption{Dataset Statistics - Enron, New Yorker, and LAMP 3U Amazon Reviews}
\label{tab:dataset_comparison}
\end{table}

\subsection{Dataset Augmentation}

\subsubsection{Synthetic Intent Generation}
We aimed to develop a fully unsupervised approach that avoids manual collection of human intents, which can be time-consuming and costly. To achieve this, we leveraged the large language model ($M_L$) to automatically extract the core content of each email or article into bullet points, emulating user input or intent.
To ensure the quality of these synthetic intents, we randomly sampled a subset and subjected them to manual human evaluation. Our findings indicated a high degree of fidelity, with over 95\% of the synthetic intents rated highly by humans.
To introduce variance and simulate different user styles, we generated three intent variants for each data point at temperatures of \texttt{0.7}, \texttt{1.0}, and \texttt{1.2}. These intents were then randomly sampled to create intent-annotated versions of our datasets. Examples of generated intents can be found in Appendix \ref{appendix:generated_intents}.

\subsubsection{Rule Generation}

To capture the nuanced stylistic preferences inherent in our datasets, we employed the large reasoning model ($M_L$) to generate natural language preference rules. These rules distill the essence of the desired style by highlighting the discrepancies between a zero-shot baseline generated by $M_L$ and the ground truth email or article.

We investigated three distinct strategies for generating these preference rules:

\begin{itemize}
    \item \textbf{Distillation-Based Rule Generation}: This approach leverages a distillation process. $M_L$ first generates a zero-shot baseline response for the given input. By analyzing the differences between this baseline and the ground truth, the model identifies missing stylistic and preference elements and generates targeted rules to bridge the gap. 
    
    \item \textbf{Direct Rule Generation}: We also explore directly prompting $M_L$ to generate rules based solely on the ground truth email or article. This approach, while simpler, lacks the targeted feedback mechanism inherent in the distillation-based method.
    
    \item \textbf{Rule Generation with Thinking Tokens}: To further enhance the rule generation process, we incorporate "thinking tokens" into the prompts. These tokens encourage the model to engage in more deliberate reasoning before generating rules, potentially leading to more insightful and effective guidance.
\end{itemize}

Examples of generated rules for each strategy are provided in Appendix~\ref{appendix:generations/rules}. A detailed discussion of the relative merits and limitations of these strategies is presented in Appendix~\ref{appendix:rulegen-strategies}.

\subsection{Preference Agent Training}

The preference agents, based on Llama-3-8B-Instruct, are trained using Quantized Low-Rank Adaptation (QLoRA) \cite{dettmers2023qlora}, a parameter-efficient fine-tuning (PeFT) method. We choose QLoRA over full fine-tuning due to its scalability and feasibility for local deployment on user devices. All model training procedures are designed to be accommodated within 16GB of VRAM, making our approach accessible to standard consumer-grade devices.

We employ a consistent set of hyperparameters across all experiments. This simplified configuration, while not optimal, serves to demonstrate the effectiveness of our method even with straightforward hyperparameter choices.  A detailed analysis of our fine-tuning procedure, including further exploration of hyperparameter search and impact on performance, can be found in Appendix \ref{appendix:finetuning-analysis}.

To establish a performance baseline, we also nai-ve finetune a model ($M_F$) using the same setup. This model is trained directly on input-output pairs (user intent and task metadata as input, ground truth text as output). This ensures a fair comparison by isolating the impact of our proposed rule-based fine-tuning approach from potential differences in model architecture or training configurations.

\subsection{Evaluation Methodology}
\label{experimental:eval-metrics}

To rigorously assess the efficacy of our preference alignment approach, we employ two complementary evaluation methodologies: automated evaluation using GPT-4 Omni (GPT-4o) and human evaluation.

\textbf{Baselines.} We compare our proposed preference agent models against several strong baselines, such as
\begin{itemize}
    \item \textbf{Zero-shot generations} from both the small model (Llama-3-8B, $M_S$) and the large model (Llama-3-70B, $M_L$),
    \item \textbf{Few-shot generations} using $M_L$, providing a limited number of examples in the prompt,
    \item \textbf{Naive fine-tuned agent} ($M_F$), where $M_S$ is directly fine-tuned on input-output pairs using QLoRA with the same hyperparameters as our preference agent training.
\end{itemize}
  

\textbf{Automated Evaluation.} We leverage GPT-4o as our automated evaluator due to its demonstrated capabilities in assessing human-written text and capturing stylistic nuances \cite{naismith-etal-2023-automated, zheng2023judging}. Our primary automated metric is the win percentage, which quantifies how often a method's output is selected by GPT-4o as the best match to the ground truth. GPT-4o's evaluations are based on criteria such as similarity in style, tone, characteristic phrases, and overall resemblance to the ground truth content.

\textbf{Human Evaluation:} We complement the automated evaluation with a human evaluation study on a subset of each dataset. Participants are presented with the original input, the ground truth output, and the outputs generated by each method. They are then asked to select the response that they believe best aligns with the ground truth, using the same criteria as the GPT-4o evaluation. This human evaluation provides valuable insights into the alignment of model outputs with human preferences. Detailed information on the human evaluation protocol can be found in Appendix \ref{appendix:human-eval}.

\textbf{Choice of Metrics.} We have deliberately chosen not to rely on traditional similarity metrics such as BLEU \cite{papineni-etal-2002-bleu} and ROUGE \cite{lin-2004-rouge}. While these metrics are valuable for assessing lexical overlap, they are less effective in capturing the nuanced aspects of stylistic similarity that are central to our evaluation. For example, consider two emails from the Enron dataset with similar content but distinct styles. One email might be written in a formal, professional tone, while the other adopts a more casual, conversational style. BLEU and ROUGE, focusing primarily on the presence or absence of specific words and phrases, might assign similar scores to both emails despite the clear difference in style perceived by a human reader. This discrepancy arises because these metrics do not adequately account for the semantic meaning and contextual usage of words, which are crucial for evaluating stylistic resemblance. This decision is further elaborated upon in Appendix \ref{appendix:automated-metrics}.

\section{Results}

We evaluated the performance of our fine-tuned preference agents against several baselines using GPT-4o and human evaluation. We report the win rates – the percentage of instances where our method outperforms the baseline – in Table \ref{table:results}.  Our baselines include zero-shot generations from both $M_S$ and $M_L$, few-shot generations using $M_L$, and a naive fine-tuned agent ($M_F$). We compare these baselines against our preference agent, trained with zero-shot baseline rules, and a no-baseline agent trained without using zero-shot information.

\begin{table*}[ht!]
    \centering
    \resizebox{\textwidth}{!}{%
        \begin{tabular}{@{}lcccccccccccc@{}}
            \toprule
            \multicolumn{1}{c}{\makecell{\textbf{Preference} \\ \textbf{Agents}}}
            & \multicolumn{3}{c}{\makecell{\textbf{New Yorker}}} 
            & \multicolumn{3}{c}{\makecell{\textbf{Enron}}} 
            & \multicolumn{3}{c}{\makecell{\textbf{LAMP 3U}}} 
            & \multicolumn{2}{c}{\makecell{\textbf{Aggregated}}} \\ 
            \cmidrule(lr){1-1}
            \cmidrule(lr){2-4} \cmidrule(lr){5-7} \cmidrule(lr){8-10} \cmidrule(lr){11-12}
            \makecell{\textbf{$M_L \to$}\\ vs $\downarrow$}
            & \makecell{\textbf{Llama3 70B} \\ \textbf{Instruct}} 
            & \makecell{\textbf{Claude 3.5} \\ \textbf{Sonnet}} 
            & \makecell{\textbf{Gemini} \\ \textbf{1.5 Pro}} 
            & \makecell{\textbf{Llama3 70B} \\ \textbf{Instruct}} 
            & \makecell{\textbf{Claude 3.5} \\ \textbf{Sonnet}} 
            & \makecell{\textbf{Gemini} \\ \textbf{1.5 Pro}} 
            & \makecell{\textbf{Llama3 70B} \\ \textbf{Instruct}} 
            & \makecell{\textbf{Claude 3.5} \\ \textbf{Sonnet}} 
            & \makecell{\textbf{Gemini} \\ \textbf{1.5 Pro}} 
            & \makecell{\textbf{LLM} \\ \textbf{Evaluation}} 
            & \makecell{\textbf{Human} \\ \textbf{Evaluation}} \\ 
            \midrule
            \makecell{Small \\ Baseline}  
            & 77.4 & 91.5 & 80.0  
            & 88.4 & 96.1 & 89.8 
            & 74.6 & 84.0 & 75.3 
            & 84.1 & 91.0 \\ 
            \midrule
            \makecell{Large \\ Baseline}  
            & 67.7 & 75.2 & 66.9 
            & 85.6 & 83.7 & 88.2 
            & 66.5 & 69.5 & 63.8 
            & 74.1 & 84.5 \\ 
            \midrule
            \makecell{Few \\ Shot}        
            & 68.3 & 62.0 & 66.7 
            & 61.1 & 68.0 & 57.4 
            & 58.3 & 57.4 & 59.4 
            & 62.0 & 73.4 \\ 
            \midrule
            \makecell{Naive \\ Finetune}  
            & 80.3 & 82.4 & 81.8 
            & 75.3 & 87.8 & 81.3 
            & 85.0 & 92.7 & 89.0 
            & 83.9 & 92.2 \\ 
            \midrule
            \makecell{No Baseline \\ Agent} 
            & 65.1 & 68.8 & 63.8 
            & 58.4 & 61.3 & 62.5 
            & 63.8 & 67.2 & 60.4 
            & 63.4 & 52.0 \\ 
            \bottomrule
        \end{tabular}
    }
    \caption{\textbf{Win Rates} of Llama3 8B $M_s$ combined with various $M_L$, evaluated by GPT4o and human evaluation.}
    \label{table:results}
\end{table*}

\textbf{Small Model Baseline.} Both GPT-4o and human evaluations agree that the small model baseline (\(M_S\)) performs significantly worse than our preference agent. This highlights the limitations of using small language models alone for tasks requiring a deep understanding of user preferences. Human evaluations show an even larger performance gap compared to GPT-4o, as humans are more adept at detecting subtle differences in style and content.

\textbf{Large Model Baseline.} While the baseline produced by the large model improves, the improvement is most noticeable in domains with lower degrees of available personalization (such as articles and product reviews), and less noticeable when direct preferences are involved (such as email writing)

\textbf{Few Shot.} While our method consistently outperforms the few-shot baseline across all datasets, the performance gap is more pronounced in the New Yorker dataset, compared to the LAMP3U or Enron datasets. We hypothesize that this difference stems from the nature of the tasks. Few-shot examples are likely more effective for email writing or product review writing, a relatively structured and concise format, than for long-form article writing, where capturing stylistic nuances requires more than a few examples. 

\textbf{Naive Finetune.} Human evaluators exhibited a stronger preference for preference agent outputs over naive fine-tuning compared to GPT-4o. Post-annotation interviews revealed that naive fine-tuning often resulted in hallucinations of crucial information, a phenomenon not effectively penalized by automated metrics but disliked by humans.

\textbf{No Baseline Agent.} The preference agent trained without access to zero-shot baseline information exhibits competitive performance, particularly when considering the marginal reduction in inference cost it offers. This makes it a viable alternative in scenarios where minimizing inference cost is a priority, even if it comes with a slight compromise in performance compared to the distillation-based approach.


Both automated and human evaluations confirm that our preference agent significantly improves the alignment of LLM outputs with individual user styles and preferences, as discussed in Section \ref{discussion:why-ft-more-effective}. The strong and consistent performance across diverse datasets and LLMs highlights the generalizability utilizing preference agents. Qualitative examples and human annotation samples of the results are provided in Appendix \ref{appendix:qualitative_examples}.

\section{Discussion}

\subsection{Enhanced Fine-Tuning through Preference Rules}
\label{discussion:why-ft-more-effective}

Our experiments (Figure \ref{fig:ruleft_vs_naiveft}) reveal that fine-tuning the preference agent on natural language rules, as opposed to directly on input-output pairs, leads to a more effective learning process. This result indicates that structured rules provide a more efficient learning signal for the preference agent.

We hypothesize that this difference stems from the inherent complexity and diversity of the target data versus the clear structure of the preference rules. When learning from raw input-output pairs, the model must adapt to the nuances of the target task, which can be challenging given the diversity of writing styles and content. Specifically, instruction-finetuned language models often exhibit a "chatbot" style, characterized by conversational and explanatory responses \cite{ouyang2022training}. This style can be significantly different from the desired output style for specific tasks like email writing, which often requires a more direct and concise approach. Adapting the model to such specific styles directly through fine-tuning can be challenging, especially under the constraints of PeFT methods. In contrast, the structured format of the rules enables the model to discern patterns more easily, facilitating faster and more effective learning. This translates to improved sample efficiency, as the model requires fewer examples to grasp the underlying preference information.

Furthermore, this approach promotes a smaller distribution shift during fine-tuning. Naive fine-tuning necessitates a substantial adaptation to the new task's distribution, which can involve shifting the model's overall output style from a conversational "chatbot" approach to a more task-specific style. PEFT methods, while effective in adapting models to new tasks or domains, may be less effective in inducing such significant changes in the model's fundamental language generation style \cite{balne2024parameter}. On the other hand, rule-based fine-tuning focuses on learning a more specific mapping – from input to preference rules – leveraging the pre-trained language model's existing capabilities for task completion. Crucially, natural language rules, are closer to the LM's existing output distribution compared to the diverse and potentially drastically different output styles of diverse, specific tasks. This makes them more suitable for PEFT adaptation, as the model can learn to generate rules without having to undergo a substantial shift in its underlying parameters.

 his decoupling of preference learning from the core task allows for more efficient adaptation, especially in multi-task settings where the model might need to switch between different domains and writing styles. By focusing on learning user preferences, we can leverage the larger model's generalizability and extensive knowledge base for superior performance across diverse tasks.

\begin{figure}
    \centering
    \includegraphics[width=1\linewidth]{resources/images/results/ruleft_vs_naiveft.png}
        \caption{On the New Yorker dataset, naive fine-tuning plateaus at a loss above 1.5, whereas fine-tuning with structured preference rules reduces the loss below 1.0 with identical hyperparameters.}
    \label{fig:ruleft_vs_naiveft}
\end{figure}

\subsection{Model-Specific Semantic Understanding}

Our findings suggest that models within the same family (e.g., Llama) exhibit a higher degree of semantic alignment compared to models from different families (e.g., GPT-4). Specifically, we observed that Llama-3 70B demonstrates a better understanding of rules generated by itself or the smaller Llama-3 8B model, compared to rules generated by GPT-4. While GPT-4 generated well-structured and seemingly comprehensive rules, they were less effective in guiding the Llama models. This indicates that semantic understanding, even when expressed through seemingly universal natural language, can be model-specific. 

This observation is further supported by experiments with human-written rules. Despite being crafted by expert annotators to be as clear and specific as possible, human-generated rules led to a 16.8\% performance degradation compared to model-generated rules. This suggests that subtle differences in the way models and humans interpret language can significantly impact the effectiveness of rule-based guidance. For instance, models might interpret terms like "precise," "concise," and "informal" differently than humans, leading to discrepancies between intended and actual outcomes.

These findings highlight the potential importance of model-specific semantic understanding in aligning LLMs with human preferences. Automated rule generation, leveraging the model's own internal representations and semantic understanding, is a more effective approach than relying on human-generated rules or prompts. However, further research is needed to fully understand the nature of these semantic differences and develop strategies for mitigating their impact.

\subsection{Enhancing Rule Generation through Deliberative Prompts}
\label{sec:discussion/thinking-tokens}

Humans often engage in a process of deliberation before formulating responses, particularly when faced with complex or nuanced tasks. This internal dialogue, where we weigh different options and consider various perspectives, contributes to more thoughtful and well-reasoned answers. Drawing inspiration from this human cognitive process, we explored the use of "deliberation" during rule generation inference by the preference agent. We include specially designed "thinking tokens," which encourage the model to engage in a similar form of internal reasoning before generating the natural language rules that guide the larger LLM. This encourages the model to decompose the task of preference extraction into smaller, more manageable steps. Our empirical results demonstrate that incorporating these deliberative prompts leads to a notable improvement in the quality of generated rules, resulting in better alignment between the large LLM's outputs and individual user preferences.

We hypothesize that these thinking tokens function as a form of cognitive scaffolding, providing the model with a structured space to isolate and process critical preference information. By explicitly prompting the model to "think" before generating rules, we aim to enhance its ability to identify subtle patterns in user preferences and translate them into effective guidance for the larger model. This approach aligns with findings from previous research, which demonstrates that prompting LLMs to engage in step-by-step reasoning can significantly improve their performance on various tasks \cite{kojima2023large, zelikman2024quietstar, goyal2024think}.  

\subsection{Evidence of Personalization}

A key objective of our approach is to learn individual writing styles rather than merely improving general task performance (e.g., email writing). To investigate this, we conducted a permutation analysis using preference agents trained on distinct email senders from the Enron dataset. We trained five agents on data from five different senders and then applied each agent to the test data of all five senders, generating emails for every agent-sender combination. This allowed us to assess whether an agent trained on a specific sender's style is more effective at generating emails resembling that sender's writing compared to other senders.

We quantified the similarity between the generated emails and the ground truth using the normalized BERT Score \cite{reimers2019sentencebert}, which provides a measure of semantic similarity suitable for analyzing large text corpora like emails. Our analysis, depicted in Figure \ref{fig:permutation}, reveals a strong trend along the diagonal. This indicates that the agent trained on a particular sender's data performs best when generating emails for that same sender, strongly suggesting that our approach successfully captures individual writing styles and preferences. 

This observation is further supported by randomly sampled human evaluations, which corroborate the BERT Score findings (see Appendix \ref{appendix:personalization} for details).

\begin{figure}
    \centering
    \includegraphics[width=\linewidth]{resources/images/results/corr2.png}
    \caption{Permutation of Models and Senders}
    \label{fig:permutation}
\end{figure}

\subsection{Cost Effectiveness}
\label{discussion:cost-effectiveness}

While our approach necessitates an inference step with \(M_L\) during rule generation and at inference time, the cost, \(C_i(M_L)\), is relatively small due to the concise nature of the rule sequences. For instance, most rule sequences generated range from 100 - 150 extra tokens. This results in a combined cost of \(C_f(M_S) + C_i(M_L)\). Although this combined cost is marginally higher than the cost of naive fine-tuning (\(C_f(M_F)\)), the significant performance gains offered by our method, as evidenced by our experimental findings, justify this trade-off. Moreover, the inference cost associated with rule generation is a one-time expense during training data preparation, further diminishing its impact.

Our decision to avoid fine-tuning \(M_L\) provides significant flexibility as we avoid the sunk cost associated with fine-tuning a large model, enabling seamless integration of newer, more powerful LLMs as they become available.

\section{Related Work}

\textbf{Traditional Methods of Alignment.} Aligning language models to human preferences often employs techniques like Reinforcement Learning from Human Feedback (RLHF) \cite{ouyang2022training} and its variant, Reinforcement Learning from AI Feedback (RLAIF) \cite{bai2022constitutional}, which leverages fine-tuned LLMs as annotators. While effective, RLHF requires substantial human annotation and complex distributed training. Direct Preference Optimization (DPO) \cite{rafailov2023direct} offers an alternative by using preference pairs, reducing computational complexity. However, DPO's reliance on contrasting pairs may not fully capture the nuances of overlapping human preferences. In-context learning methods \cite{NEURIPS2022_8bb0d291, woźniak2024personalized}, while showing promise, are limited by context length restrictions, hindering their ability to generalize effectively.

\textbf{Agent-based Alignment.} To address the computational demands of training large models, agent-based architectures have emerged as a promising avenue for compute-constrained environments. For instance, \citet{li2023guiding} utilize a fine-tuned T5 policy model to guide large models via stimulus prompting. However, this approach necessitates full-parameter Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) optimization, introducing computational overhead and yielding limited performance gains in tasks like dialogue generation. Similarly, Aligner \cite{ji2024aligner} employs full-parameter SFT and relies on a substantial custom dataset for preference learning, posing challenges in terms of data requirements and VRAM usage. \citet{tan2024democratizing} propose Parameter-Efficient Fine-Tuning (PEFT) methods to personalize agents based on user history and preference retrieval. While computationally efficient, this approach is constrained by the reasoning capabilities of the smaller fine-tuned agent. These approaches often rely on automatic metrics like BLEU and ROUGE, which predominantly capture lexical similarity without fully encapsulating the nuances of human preferences. \citet{gao2024aligning} introduce an agent trained on human edits to align zero-shot outputs. However, this approach requires multiple inference rounds for each query, increasing latency and computational costs. Moreover, human edit history may not consistently reflect genuine preferences, and relying solely on edit distance as a measure of alignment can be unreliable. \citet{yang2024aligning} propose a framework for aligning LLMs through Multi-perspective User Preference Ranking-based Feedback. This approach, however, involves an initial SFT phase, along with Multi-Perspective Ranking Aggregation (MPRA) and Reward Imitation Learning (RIL), leading to significant training overhead and the use of metrics like BLEU that may not accurately capture human preferences.

\textbf{Comparison with Aligner.} While both Aligner \cite{ji2024aligner} and our method utilize a small model trained with full-parameter SFT, our approaches differ significantly. Aligner focuses on correcting model outputs post-generation, while our preference agent proactively generates rules to steer the large model's initial output. This allows us to leverage the large model's reasoning capabilities by providing preference information upfront, rather than correcting its output afterwards. While Aligner demonstrates strong performance on tasks like text summarization and dialogue generation, its design is geared towards making smaller adjustments to large model outputs. Our task, on the other hand, often requires more substantial changes to align with user preferences, potentially necessitating complete rewrites of emails or articles. An Aligner-style approach or naive fine-tuning would face challenges in our setting, as a small model might struggle to accurately make drastic changes while preserving the large model's knowledge and coherence. This would also complicate the fine-tuning objective, as the patterns to be learned would be less well-defined and vary significantly across examples.

\section{Conclusion}

This work introduces a novel paradigm for aligning large language models (LLMs) with individual user preferences using limited data. We leverage small, locally trainable "preference agents" to efficiently guide larger LLMs without resource-intensive fine-tuning. Our approach generates natural language rules that encapsulate user preferences, acting as a "steering wheel" to direct the larger model's output towards desired styles and content.

This framework introduces a new preference fine-tuning objective: learning from implicit preference information found in the differences between a baseline LLM output and the user's desired output. This allows the agent to distill user preferences into actionable rules, enabling efficient personalization without modifying the larger model's weights.

Our empirical findings across diverse datasets demonstrate that preference agents significantly improve alignment with user preferences compared to existing methods in a compute-efficient manner, highlighting the potential for building highly personalized LLM applications at scale.

\section*{Limitations}

While our proposed method demonstrates significant improvements, there are a few areas for potential refinement. One consideration is the time required for the large model to process the preference agent's output before the first token can be generated. This could lead to a slightly higher Time to First Token (TTFT) at inference time. However, we believe the substantial performance gains offered by our approach outweigh this trade-off. 

As discussed in \S\ref{appendix:rulegen-strategies}, our most performant rule generation strategy incurs an additional computational cost compared to the alternative methods due to an extra zero-shot inference step. This cost is offset by the superior performance it enables.  We also provide a highly competitive "no-baseline" rule generation method which offers good performance at a lower inference cost. 

Furthermore, our rule generation strategy leverages thinking tokens, which can lead to slightly longer outputs. If output length is a strict constraint, this step can be omitted with minimal impact on the framework's effectiveness. Importantly, the inference cost associated with rule generation is a one-time expense incurred during training data preparation. 

Finally, as noted in \S\ref{discussion:cost-effectiveness}, using $M_L$ for preference agent rule generation introduces an additional inference iteration compared to naive fine-tuning.

While our current research focuses on text-based preferences, future work could explore extending this approach to other modalities, such as image or audio generation.  Additionally, investigating the integration of multimodal preferences and the development of more sophisticated rule generation strategies could further enhance the capabilities of preference agents. We believe that this research opens exciting new avenues for personalized LLM applications, paving the way for a future where powerful language models can be seamlessly tailored to individual needs and preferences, ultimately enhancing user experience and fostering more engaging and human-centric interactions. 

\section*{Ethical Considerations}

In this work, we have taken several steps to ensure that our research adheres to ethical principles and respects the rights of all parties involved. We are committed to the responsible and ethical use of AI technology and have implemented measures to prevent potential misuse of our work.

\paragraph{Dataset Licensing and Attribution.} Both datasets used in this research will be released under the Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) license.

The Enron email dataset \cite{10.1007/978-3-540-30115-8_22} is available for educational and research purposes under the principles of fair use. We have credited the original dataset creators and adhered to the terms of its usage.

The New Yorker dataset is based on the 'All the News 2.0' dataset by Andrew Thompson \cite{allthenews2}, which is licensed for non-commercial, research purposes only. We have made modifications and enhancements to the dataset, and these changes are also licensed under the CC BY-NC 4.0 license. We have properly attributed the original dataset and its creator.

\paragraph{Model Release.} In compliance with the terms of the 'All the News 2.0' dataset license, we will not be releasing the fine-tuned agents trained on the New Yorker dataset. The license explicitly states that the dataset is to be used for research purposes only and not for the release of commercial generative models.

Similarly, we will not release the agent fine-tuned on the Enron email corpus. This decision was made to ensure that our models are not used to impersonate the senders in the Enron email corpus without their explicit permission. We believe that releasing such a model could potentially infringe upon the privacy rights of the individuals involved. 

However, for research purposes only, we will make the models available upon request.

\paragraph{Citation and Acknowledgment.} We have taken extensive care to ensure that we comply with all licenses and have appropriately cited any of our work that is a derivative of another project. We acknowledge the original creators and their contributions to the field.

\paragraph{Potential Misuse.} We acknowledge that our datasets, though open-source, can potentially be used to train AI assistants or models for malicious purposes. We strongly condemn any misuse of our work and explicitly support the safe and responsible use of AI technology. Our intention is to advance the field of AI research while adhering to ethical principles and preventing harm.


\section*{Acknowledgements}

We would like to express our sincere gratitude to the reviewers of ACL ARR for their insightful and constructive feedback, which significantly improved the quality of this paper. We are particularly grateful for their suggestion to incorporate the LAMP dataset, which broadened the scope and impact of our evaluation.