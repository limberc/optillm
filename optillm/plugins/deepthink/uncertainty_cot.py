import logging
import re
from collections import Counter
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional
from .utils import create_completion

logger = logging.getLogger(__name__)


class UncertaintyRoutedCoT:
    def __init__(self, client, model: str, max_tokens: int = 16382, request_config: Optional[Dict[str, Any]] = None):
        self.client = client
        self.model = model
        if request_config:
            self.max_tokens = request_config.get('max_completion_tokens') or request_config.get('max_tokens', max_tokens)
        else:
            self.max_tokens = max_tokens
        self.completion_tokens = 0

    def generate_with_uncertainty_routing(
            self,
            prompt: str,
            num_samples: int = 3,
            confidence_threshold: float = 0.7,
            temperature: float = 0.7,
            top_p: float = 0.95
    ) -> Dict[str, Any]:
        logger.info(f"Generating {num_samples} samples for uncertainty routing")

        samples = self._generate_multiple_samples(prompt, num_samples, temperature, top_p)
        greedy_sample = self._generate_greedy_sample(prompt)

        sample_data = []
        for sample in samples:
            thinking = self._extract_thinking(sample)
            answer = self._extract_answer(sample)
            sample_data.append({
                "full_response": sample,
                "thinking": thinking,
                "answer": answer
            })

        greedy_thinking = self._extract_thinking(greedy_sample)
        greedy_answer = self._extract_answer(greedy_sample)

        confidence_score = self._evaluate_confidence(sample_data)

        logger.debug(f"Confidence: {confidence_score:.3f}, Samples: {[s['answer'][:50] for s in sample_data if s['answer']]}")

        if confidence_score >= confidence_threshold:
            final_response = self._majority_vote_response(sample_data)
            routing_decision = "majority_vote"
            logger.info(f"High confidence ({confidence_score:.3f}) - majority vote")
        else:
            final_response = greedy_sample
            routing_decision = "greedy"
            logger.info(f"Low confidence ({confidence_score:.3f}) - greedy sample")

        return {
            "final_response": final_response,
            "confidence_score": confidence_score,
            "routing_decision": routing_decision,
            "samples": sample_data,
            "greedy_sample": {
                "full_response": greedy_sample,
                "thinking": greedy_thinking,
                "answer": greedy_answer
            },
            "completion_tokens": self.completion_tokens
        }

    def _generate_multiple_samples(self, prompt: str, num_samples: int, temperature: float, top_p: float) -> List[str]:
        samples = []
        for i in range(num_samples):
            logger.debug(f"Generating sample {i + 1}/{num_samples}")
            response = create_completion(
                client=self.client,
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.max_tokens,
                temperature=temperature,
                top_p=top_p
            )
            self.completion_tokens += response.usage.completion_tokens
            if not response.choices or response.choices[0].message.content is None:
                logger.warning(f"Sample {i + 1} empty")
                samples.append("")
            else:
                samples.append(response.choices[0].message.content.strip())
        return samples

    def _generate_greedy_sample(self, prompt: str) -> str:
        logger.debug("Generating greedy sample")
        response = create_completion(
            client=self.client,
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            temperature=0.0
        )
        self.completion_tokens += response.usage.completion_tokens
        if not response.choices or response.choices[0].message.content is None:
            logger.error("Greedy sample truncated")
            return "Error: Response truncated due to token limit."
        return response.choices[0].message.content.strip()

    def _extract_thinking(self, response: str) -> str:
        match = re.search(r'<think>(.*?)</think>', response, re.DOTALL)
        return match.group(1).strip() if match else ""

    def _extract_answer(self, response: str) -> str:
        if not response:
            return ""
        think_end = response.find('</think>')
        answer_part = response[think_end + 8:].strip() if think_end != -1 else response.strip()
        if not answer_part:
            return ""

        patterns = [
            r'(?:the )?(?:final )?answer is:?\s*(.+?)(?:\n|$)',
            r'(?:therefore|thus|so|hence),?\s*(?:the )?(?:answer is:?\s*)?(.+?)(?:\n|$)',
            r'(?:conclusion|result|solution):?\s*(.+?)(?:\n|$)',
            r'\\boxed\{([^}]+)\}',
            r'(?:equals?|=?)\s*([\d\w\.\-\+\/\*\^\(\)]+?)(?:\n|$)',
        ]

        for pattern in patterns:
            try:
                match = re.search(pattern, answer_part, re.IGNORECASE)
                if match and match.group(1).strip():
                    return self._normalize_answer(match.group(1).strip())
            except Exception:
                continue

        for line in answer_part.split('\n'):
            line = line.strip()
            if line and len(line) > 10:
                return self._normalize_answer(line)

        return self._normalize_answer(answer_part[:200])

    def _normalize_answer(self, answer: str) -> str:
        if not answer:
            return ""
        normalized = ' '.join(answer.split()).lower()
        normalized = re.sub(r'[^\w\s\.\-\+\*/\^=]', '', normalized)
        number_words = {'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4',
                       'five': '5', 'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10'}
        for word, num in number_words.items():
            normalized = re.sub(r'\b' + word + r'\b', num, normalized, flags=re.IGNORECASE)
        return normalized.strip('.,;:!?')

    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        if not text1 or not text2:
            return 0.0
        base_similarity = SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
        nums1 = re.findall(r'[\d\.]+', text1)
        nums2 = re.findall(r'[\d\.]+', text2)
        if nums1 and nums2:
            try:
                if set(float(n) for n in nums1) == set(float(n) for n in nums2):
                    return max(base_similarity, 0.9)
            except ValueError:
                pass
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if words1 and words2:
            overlap = len(words1 & words2) / min(len(words1), len(words2))
            return max(base_similarity, overlap * 0.8)
        return base_similarity

    def _evaluate_confidence(self, sample_data: List[Dict[str, Any]]) -> float:
        if len(sample_data) < 2:
            return 0.5
        answers = [s["answer"] for s in sample_data if s["answer"]]
        thinking_texts = [s["thinking"] for s in sample_data if s["thinking"]]
        if not answers:
            return 0.1
        answer_consistency = self._calculate_answer_consistency(answers)
        reasoning_consistency = self._calculate_reasoning_consistency(thinking_texts)
        confidence = 0.6 * answer_consistency + 0.4 * reasoning_consistency
        logger.debug(f"Consistency: ans={answer_consistency:.3f}, reason={reasoning_consistency:.3f}, total={confidence:.3f}")
        return confidence

    def _calculate_answer_consistency(self, answers: List[str]) -> float:
        if len(answers) < 2:
            return 0.5
        normalized = [self._normalize_answer(a) for a in answers]
        counts = Counter(normalized)
        most_common = counts.most_common(1)[0][1]
        agreement = most_common / len(answers)
        max_sim = max((SequenceMatcher(None, a1, a2).ratio() 
                      for i, a1 in enumerate(normalized) for a2 in normalized[i+1:]), default=0)
        return min(max(agreement, max_sim), 1.0)

    def _calculate_reasoning_consistency(self, texts: List[str]) -> float:
        if len(texts) < 2:
            return 0.5
        texts_lower = [t.lower() for t in texts]
        sims = [SequenceMatcher(None, t1, t2).ratio() 
               for i, t1 in enumerate(texts_lower) for t2 in texts_lower[i+1:]]
        return min(sum(sims) / len(sims), 1.0) if sims else 0.5

    def _majority_vote_response(self, sample_data: List[Dict[str, Any]]) -> str:
        answers = [s["answer"] for s in sample_data if s["answer"]]
        if not answers:
            return sample_data[0]["full_response"]
        normalized = [self._normalize_answer(a) for a in answers]
        most_common = Counter(normalized).most_common(1)[0][0]
        best = max((s for s in sample_data if s["answer"] and self._normalize_answer(s["answer"]) == most_common),
                  key=lambda s: len(s["thinking"]), default=None)
        return best["full_response"] if best else sample_data[0]["full_response"]
