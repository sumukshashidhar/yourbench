# yourbench/pipeline/multi_hop_question_generation.py

"""
Multi-Hop Question Generation Module

Minimal approach:
----------------
- Each row in the dataset has "multihop_chunks", which is a list of dicts:
    { "chunk_ids": [...], "chunks_text": [...] }
- For each such multi-hop item, we generate questions that may require
  combining multiple single-hop chunks. We do NOT assign a new multi-hop ID.
- In the final question dataset, we store a list of source_chunk_ids that
  show which single-hop chunks were used. We do not store chunk_uuid or any
  location indices. The final row is basically:
  [
    {
      "document_id": ...,
      "source_chunk_ids": [...],
      "question": ...,
      "self_answer": ...,
      ...
    },
    ...
  ]
"""

import json
import re
from dataclasses import dataclass, field
from typing import Dict, Any, List

from loguru import logger
from datasets import Dataset

from yourbench.utils.dataset_engine import smart_load_dataset
from yourbench.utils.dataset_engine import save_dataset
from yourbench.utils.inference_engine import InferenceCall, run_inference
from yourbench.utils.prompts import (
    MULTI_HOP_QUESTION_GENERATION_SYSTEM_PROMPT,
    MULTI_HOP_QUESTION_GENERATION_USER_PROMPT,
)


@dataclass
class MultiHopQuestionRow:
    """
    Minimal structure for multi-hop question rows:
      - document_id: which doc
      - source_chunk_ids: which single-hop chunks are used
      - question, self_answer, etc.
    """
    document_id: str
    source_chunk_ids: List[str]

    question: str
    self_answer: str
    estimated_difficulty: int
    self_assessed_question_type: str
    generating_model: str
    thought_process: str
    citations: List[str] = field(default_factory=list)


def run(config: Dict[str, Any]) -> None:
    try:
        stage_cfg = config.get("pipeline", {}).get("multi_hop_question_generation", {})
        if not stage_cfg.get("run", False):
            logger.info("multi_hop_question_generation stage is disabled. Skipping.")
            return

        # Add defensive error handling throughout entire function
        source_dataset_name = stage_cfg.get("source_dataset_name")
        if not source_dataset_name:
            logger.error("No source_dataset_name provided in config. Skipping multi_hop_question_generation.")
            return
            
        output_dataset_name = stage_cfg.get("output_dataset_name")
        if not output_dataset_name:
            logger.error("No output_dataset_name provided in config. Skipping multi_hop_question_generation.")
            return
            
        logger.info("Loading chunked dataset: {}", source_dataset_name)
        dataset = smart_load_dataset(source_dataset_name, config)
        logger.info("Loaded dataset with {} rows.", len(dataset))

        system_msg = {"role": "system", "content": MULTI_HOP_QUESTION_GENERATION_SYSTEM_PROMPT}
        all_inference_calls: List[InferenceCall] = []
        call_index_map: List[tuple] = []

        for row_idx, row in enumerate(dataset):
            doc_summary = row.get("document_summary", "No summary provided.")
            title = row.get("document_filename", f"Document_{row_idx}")
            doc_id = row.get("document_id", f"doc_{row_idx}")

            multi_hop_chunks = row.get("multihop_chunks", [])
            if not isinstance(multi_hop_chunks, list) or not multi_hop_chunks:
                continue

            for mh_idx, mh in enumerate(multi_hop_chunks):
                if not isinstance(mh, dict):
                    continue

                subchunk_ids = mh.get("chunk_ids", [])
                subchunk_texts = mh.get("chunks_text", [])
                if not subchunk_texts:
                    continue

                # Build multiple <text_chunk_i> tags
                text_chunks_aggregated = ""
                for i, sc_text in enumerate(subchunk_texts):
                    text_chunks_aggregated += f"<text_chunk_{i}>{sc_text}</text_chunk_{i}>\n"

                additional_instructions = stage_cfg.get("additional_instructions", "undergraduate")
                user_prompt = MULTI_HOP_QUESTION_GENERATION_USER_PROMPT.format(
                    title=title,
                    document_summary=doc_summary,
                    chunks=text_chunks_aggregated,
                    additional_instructions=additional_instructions
                )
                user_msg = {"role": "user", "content": user_prompt}
                inference_call = InferenceCall(
                    messages=[system_msg, user_msg],
                    tags=["multi_hop_qa"]
                )
                all_inference_calls.append(inference_call)
                call_index_map.append((row_idx, doc_id, subchunk_ids))

        if not all_inference_calls:
            logger.warning("No multi-hop chunks found. Exiting multi-hop question generation.")
            return

        logger.info("Sending {} calls to inference for multi-hop question generation.", len(all_inference_calls))
        responses_dict = run_inference(
            config=config,
            step_name="multi_hop_question_generation",
            inference_calls=all_inference_calls,
        )

        question_dataset_rows: List[Dict[str, Any]] = []

        for model_name, model_responses in responses_dict.items():
            logger.info("Processing {} responses for model: {}", len(model_responses), model_name)
            if len(model_responses) != len(call_index_map):
                logger.error(
                    "Model '{}' returned {} responses, expected {}. Possibly mismatch or truncation.",
                    model_name, len(model_responses), len(call_index_map)
                )

            for idx, raw_resp in enumerate(model_responses):
                if idx >= len(call_index_map):
                    break
                row_idx, doc_id, source_chunk_ids = call_index_map[idx]

                json_str = _extract_output_json(raw_resp)
                if not json_str.strip():
                    logger.warning("No parseable JSON for row={}, doc_id={} (model={}).", row_idx, doc_id, model_name)
                    continue

                try:
                    question_answer_pairs = json.loads(json_str)
                except Exception as e:
                    logger.warning("Failed to parse JSON for row={}, doc_id={} (model={}): {}", row_idx, doc_id, model_name, e)
                    continue

                if not isinstance(question_answer_pairs, list):
                    logger.warning("JSON is not a list for row={}, doc_id={} (model={}).", row_idx, doc_id, model_name)
                    continue

                for qap in question_answer_pairs:
                    try:
                        # Extract and validate question and answer
                        question = qap.get("question", "")
                        if not question.strip():
                            logger.warning(f"Empty question for row={row_idx}, doc_id={doc_id}, skipping")
                            continue
                        
                        self_answer = qap.get("answer", "")
                        
                        # Ensure difficulty is an integer
                        difficulty_raw = qap.get("estimated_difficulty", 5)
                        try:
                            difficulty = int(difficulty_raw)
                        except (ValueError, TypeError):
                            logger.warning(f"Invalid difficulty value '{difficulty_raw}' for doc_id={doc_id}, defaulting to 5")
                            difficulty = 5
                        
                        # Validate and extract other fields
                        qtype = qap.get("question_type", "unknown")
                        if not isinstance(qtype, str):
                            logger.warning(f"Invalid question_type value '{qtype}' for doc_id={doc_id}, converting to string")
                            qtype = str(qtype)
                        
                        thought_process = qap.get("thought_process", "")
                        if not isinstance(thought_process, str):
                            thought_process = str(thought_process) if thought_process is not None else ""
                        
                        # Ensure citations is a list
                        cits = qap.get("citations", [])
                        if not isinstance(cits, list):
                            logger.warning(f"Citations is not a list for doc_id={doc_id}, converting to list")
                            if cits is None:
                                cits = []
                            else:
                                try:
                                    if isinstance(cits, str) and cits.strip().startswith("[") and cits.strip().endswith("]"):
                                        cits = json.loads(cits)
                                    else:
                                        cits = [cits]
                                except:
                                    cits = [str(cits)]
                        
                        # Create and append row object
                        row_obj = MultiHopQuestionRow(
                            document_id=doc_id,
                            source_chunk_ids=source_chunk_ids,
                            question=question,
                            self_answer=self_answer,
                            estimated_difficulty=difficulty,
                            self_assessed_question_type=qtype,
                            generating_model=model_name,
                            thought_process=thought_process,
                            citations=cits
                        )
                        question_dataset_rows.append(row_obj.__dict__)
                    except Exception as e:
                        logger.warning(f"Error processing QA pair for doc_id={doc_id}: {str(e)}")
                        # Skip this pair but continue processing others
                        continue

        if not question_dataset_rows:
            logger.warning("No valid question rows produced from multi-hop generation.")
            return

        logger.info("Constructing multi-hop question-level dataset with {} rows...", len(question_dataset_rows))
        
        # Prepare data with explicit type handling to prevent Arrow conversion errors
        dataset_dict = {}
        for key in question_dataset_rows[0].keys():
            try:
                # Extract all values for this column
                values = [d.get(key) for d in question_dataset_rows]
                
                # Handle specific field types
                if key == 'estimated_difficulty':
                    # Ensure all difficulty values are integers
                    cleaned_values = []
                    for val in values:
                        try:
                            cleaned_values.append(int(val))
                        except (ValueError, TypeError):
                            logger.warning(f"Invalid {key} value: {val}, defaulting to 5")
                            cleaned_values.append(5)
                    dataset_dict[key] = cleaned_values
                elif key == 'source_chunk_ids' or key == 'citations':
                    # Ensure list fields are actually lists
                    cleaned_values = []
                    for val in values:
                        if isinstance(val, list):
                            cleaned_values.append(val)
                        elif val is None:
                            cleaned_values.append([])
                        else:
                            try:
                                # Try to convert string representation to list if needed
                                if isinstance(val, str) and (val.startswith('[') and val.endswith(']')):
                                    cleaned_val = json.loads(val)
                                    if isinstance(cleaned_val, list):
                                        cleaned_values.append(cleaned_val)
                                    else:
                                        cleaned_values.append([])
                                else:
                                    cleaned_values.append([val])
                            except:
                                logger.warning(f"Invalid {key} value: {val}, defaulting to empty list")
                                cleaned_values.append([])
                    dataset_dict[key] = cleaned_values
                else:
                    # For other fields, ensure strings are strings
                    cleaned_values = []
                    for val in values:
                        if val is None:
                            if key in ['question', 'self_answer', 'self_assessed_question_type', 'thought_process']:
                                cleaned_values.append("")
                            else:
                                cleaned_values.append(val)
                        elif isinstance(val, (str, int, float, bool)):
                            cleaned_values.append(val)
                        else:
                            try:
                                # Convert to string if not a primitive type
                                cleaned_values.append(str(val))
                            except:
                                logger.warning(f"Invalid {key} value: {val}, defaulting to empty string")
                                cleaned_values.append("")
                    dataset_dict[key] = cleaned_values
            except Exception as e:
                logger.error(f"Error processing column {key}: {str(e)}")
                # Skip this column rather than failing the whole pipeline
                continue
        
        try:
            question_dataset = Dataset.from_dict(dataset_dict)
        except Exception as e:
            logger.error(f"Failed to create dataset from dictionary: {str(e)}")
            # Try a more defensive approach with explicit column creation
            try:
                # Start with required minimal columns
                min_dict = {
                    "document_id": [d.get("document_id", "unknown") for d in question_dataset_rows],
                    "question": [d.get("question", "") for d in question_dataset_rows],
                    "self_answer": [d.get("self_answer", "") for d in question_dataset_rows]
                }
                question_dataset = Dataset.from_dict(min_dict)
                logger.warning("Created minimal dataset with only essential columns due to conversion errors")
            except Exception as e2:
                logger.critical(f"Cannot create even minimal dataset: {str(e2)}")
                # Return with error rather than crash the pipeline
                logger.warning("Multi-hop question generation completed with errors - no dataset created.")
                return

        logger.info("Saving multi-hop question dataset as '{}'.", output_dataset_name)
        save_dataset(
            dataset=question_dataset,
            step_name="multi_hop_question_generation",
            config=config,
            output_dataset_name=output_dataset_name
        )
        logger.success("Multi-hop question generation completed successfully.")
    except Exception as e:
        logger.error(f"Error in run function: {str(e)}")
        logger.warning("Multi-hop question generation completed with errors - no dataset created.")


def _extract_tag_content(text: str, tag: str) -> str:
    pattern = fr"<{tag}\s*>([\s\S]*?)</{tag}>"
    m = re.search(pattern, text)
    return m.group(1).strip() if m else ""


def _extract_output_json(raw_response: str) -> str:
    try:
        # 1. Check <output_json> block
        extracted = _extract_tag_content(raw_response, "output_json")
        if extracted.strip():
            sanitized = _maybe_strip_triple_backticks(extracted)
            if sanitized.strip():
                return sanitized

        # 2. Check ```json fenced block
        fence_pattern = r"```json\s*([\s\S]*?)\s*```"
        fm = re.search(fence_pattern, raw_response)
        if fm:
            return fm.group(1).strip()

        # 3. Check just ``` block (without json qualifier)
        simple_fence_pattern = r"```\s*([\s\S]*?)\s*```"
        sfm = re.search(simple_fence_pattern, raw_response)
        if sfm:
            return sfm.group(1).strip()

        # 4. fallback bracket extraction
        cands = _best_effort_json_extract(raw_response)
        if cands:
            # Try to validate each candidate as valid JSON
            for cand in cands:
                try:
                    # Quick check if it could be valid JSON
                    if (cand.startswith("[") and cand.endswith("]")) or (cand.startswith("{") and cand.endswith("}")):
                        json.loads(cand)  # Will raise exception if invalid
                        return cand  # Return the first valid JSON
                except:
                    continue  # Try next candidate
            
            # If we got here, none of the candidates validated, return the first one anyway
            return cands[0]
            
        # 5. Final fallback: try to extract anything that looks like a list of questions
        q_pattern = r'"question"\s*:\s*"([^"]+)"'
        q_matches = re.findall(q_pattern, raw_response)
        if q_matches:
            logger.warning("Using very aggressive JSON extraction fallback")
            # Construct a minimal JSON array with just the questions
            questions_json = []
            for q in q_matches:
                questions_json.append({"question": q, "answer": "Extraction fallback", "estimated_difficulty": 5, "question_type": "unknown"})
            return json.dumps(questions_json)
            
        return ""
    except Exception as e:
        logger.warning(f"Error in _extract_output_json: {str(e)}")
        return ""


def _maybe_strip_triple_backticks(text_in: str) -> str:
    pattern = r"^\s*```(?:json)?\s*([\s\S]*?)\s*```$"
    mm = re.match(pattern, text_in)
    if mm:
        return mm.group(1)
    return text_in


def _best_effort_json_extract(full_text: str) -> List[str]:
    pattern = r"([\[{].*?[\]}])"
    matches = re.findall(pattern, full_text, flags=re.DOTALL)
    cands = []
    for m in matches:
        if (m.startswith("[") and m.endswith("]")) or (m.startswith("{") and m.endswith("}")):
            cands.append(m.strip())
    return cands
