import re
from collections import Counter
from typing import Any, Dict, List, Optional, Union

from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    load_dataset,
)

from treetune import logging_utils
from treetune.tasks import Task
from treetune.tokenization_utils import Tokenizer

logger = logging_utils.get_logger(__name__)


@Task.register("countdown", exist_ok=True)
class Countdown(Task):
    def __init__(
        self,
        prepend_in_context_few_shot: bool,
        use_original_format: bool = False,
        remove_calculator_expressions: bool = True,
        intermediate_step_delimiter: Optional[str] = "\n",
        answer_prefix: Optional[str] = None, 
        ensure_fit_in_context_size: bool = False,
        max_few_shot_dataset_size: Optional[int] = None,
        context_size: Optional[int] = None,
        max_generation_tokens: Optional[int] = None,
        inplace_split_solution: bool = False,
        few_shot_dataset_path: Optional[str] = None,
        use_minerva_few_shot_prompt: bool = False,
        use_gold_steps_for_few_shot: bool = False,
        num_few_shot_examples: Optional[int] = None,
        max_few_shot_problem_length: Optional[int] = None,
        max_few_shot_solution_length: Optional[int] = None,
        max_few_shot_num_steps: Optional[int] = None,
        tokenizer: Optional[Tokenizer] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)


    def extract_predicted_answer_from_text(
        self, text: str, problem: Optional[str] = None
    ) -> Optional[str]:
        pass

    def extract_gold_answer_from_text(self, text: str) -> str:
        pass

    def split_solution_into_intermediate_steps(self, solution: str) -> List[int]:
        """
        Split the solution into reasoning steps.

        Args:
            solution: The solution text.

        Returns:
            A list of indices where each index corresponds to the start of a reasoning step.
            Example:
            >>> solution = '...'
            >>> indices = split_solution_into_reasoning_steps(solution)
            >>> steps = [solution[indices[i]:indices[i+1]] for i in range(len(indices) - 1)]
        """
        assert self.use_original_format, "This method is only for original format"
        assert self.intermediate_step_delimiter is not None
        pass

    def verify_answer(self, target: int, nums: str, answer: str) -> bool:
        # answer is a sequence of operations
        # written in the format Step 1: 1+2=3\nStep 2: 3*3=9, etc.
        try:
            nums = nums.split("using the numbers")[1].strip()
            nums = nums.split(". Use")[0].strip()
            nums = [int(num.strip()) for num in nums.split(",")]
            print(nums, target)
            answer = answer.lower().strip()
            steps = answer.split("step")
            print(steps)
            parsed_steps = []
            # check if all steps are valid
            for s, step in enumerate(steps):
                if ":" not in step:
                    continue
                step = step.strip()
                try:
                    step = step.split(":")[1]
                    parsed_steps.append(step)
                    lhs, rhs = step.split("=")
                    lhs_answer = eval(lhs)
                    rhs_answer = eval(rhs)
                    if lhs_answer != rhs_answer:
                        print(f"Step {s+1} {lhs} != {rhs}")
                        return False
                except Exception as e:
                    print(f"Error in step {s+1}: {e}")
                    return False
                if s == len(steps) - 1:
                    if lhs_answer != target:
                        print(f"Last step {lhs_answer} != {target}")
                        return False
            print(parsed_steps)
            nums_to_use = nums.copy()
            # check if all numbers are used
            for s, step in enumerate(parsed_steps):
                # step is of the format 1+2=3
                lhs, rhs = step.split("=")
                rhs = float(rhs.strip())
                nums_to_use.append(rhs)
                if '+' in lhs:
                    a1, a2 = lhs.split('+')
                    a1, a2 = float(a1.strip()), float(a2.strip())
                    if a1 not in nums_to_use or a2 not in nums_to_use:
                        print(f"Step {s+1} {lhs} not in nums_to_use")
                        return False
                    if a1 == a2:
                        if nums_to_use.count(a1) != 2:
                            return False
                    nums_to_use.remove(a1)
                    nums_to_use.remove(a2)
                elif '-' in lhs:
                    a1, a2 = lhs.split('-')
                    a1, a2 = float(a1.strip()), float(a2.strip())
                    if a1 not in nums_to_use or a2 not in nums_to_use:
                        print(f"Step {s+1} {lhs} not in nums_to_use")
                        return False
                    if a1 == a2:
                        if nums_to_use.count(a1) != 2:
                            return False
                    nums_to_use.remove(a1)
                    nums_to_use.remove(a2)
                elif '*' in lhs:
                    a1, a2 = lhs.split('*')
                    a1, a2 = float(a1.strip()), float(a2.strip())
                    if a1 not in nums_to_use or a2 not in nums_to_use:
                        print(f"Step {s+1} {lhs} not in nums_to_use")
                        return False
                    if a1 == a2:
                        if nums_to_use.count(a1) != 2:
                            print(f"Step {s+1} {lhs} {a1} not used twice")
                            return False
                    nums_to_use.remove(a1)
                    nums_to_use.remove(a2)
                elif '/' in lhs:
                    a1, a2 = lhs.split('/')
                    a1, a2 = int(a1.strip()), int(a2.strip())
                    if a1 not in nums_to_use or a2 not in nums_to_use:
                        print(f"Step {s+1} {lhs} not in nums_to_use")
                        return False
                    if a1 == a2:
                        if nums_to_use.count(a1) != 2:
                            return False
                    nums_to_use.remove(a1)
                    nums_to_use.remove(a2)
                else:
                    print(f"Step {s+1} {lhs} no operation found")
                    return False
            if len(nums_to_use) != 1:
                print(f"Not all numbers used: {nums_to_use}")
                return False
            if nums_to_use[0] != target:
                print(f"Last number {nums_to_use[0]} != {target}")
                return False
        except Exception as e:
            print(f"Error in verify_answer: {e}")
            return False
        return True

    def evaluate_predictions(
        self,
        *,
        predictions: List[List[str]] = None,
        references: Dataset = None,
    ) -> Dict[str, float]:
        once_hit_acc = []
        correct_frac = []
        majority_vote_acc = []
        unique_answer_count = []
        none_answer_extracted = []

        for solution_candidates, ref in zip(predictions, references):
            problem = ref["problem"]
            target = ref["target"]
            assert len(solution_candidates) > 0

            grading_results = []
            for solution in solution_candidates:
                result = self.verify_answer(
                    target=target, nums=problem, answer=solution
                )
                grading_results.append(result)

            once_hit_acc.append(float(any(grading_results)))
            correct_frac.append(sum(grading_results) / len(grading_results))


        once_hit = sum(once_hit_acc) / len(once_hit_acc)
        correct_frac = sum(correct_frac) / len(correct_frac)

        return {
            "once_hit": once_hit,
            "exact_match": once_hit,  # for backwards compatibility
            "correct_frac": correct_frac,
            "exact_match_frac": correct_frac,  # for backwards compatibility
            "majority_vote_acc": 0,
            "unique_answer_count": 0,
            "none_answer_extracted_frac_per_problem": (
                0
            ),
        }

    def build_dataset(
        self,
    ) -> DatasetDict:
        datasets = super().build_dataset()
        datasets = datasets.map(
            self._preprocess_example, num_proc=4, desc="Preprocessing examples"
        )
        return datasets

    def _preprocess_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        question = example["input"].strip()
        target = example["target"]
        output = {}
        output.update({
            "problem": question,
            "target": target,
        })
        return output
    
    def _load_data_source(
        self,
    ) -> Union[DatasetDict, Dataset, IterableDatasetDict, IterableDataset]:
        dataset = load_dataset("json", data_files={"train": "data/cd_train.json", "validation": "data/cd_val.json", "test": "data/cd_test.json"})
        return dataset

