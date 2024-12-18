import itertools
import random
from typing import Dict, List

def combine_nums(a, b):
    # Implicitly makes assumptions about the order of operations and valid operations
    a = int(a)
    b = int(b)
    possible = [[a+b, f"{a}+{b}={a+b}"], [a*b, f"{a}*{b}={a*b}"]]
    if a <= b:
        possible.append([b-a, f"{b}-{a}={b-a}"])
        if a != 0 and b % a == 0:
            possible.append([b//a, f"{b}/{a}={round(b//a,0)}"])
    else:
        possible.append([a-b, f"{a}-{b}={a-b}"])
        if b != 0 and a % b == 0:
            possible.append([a//b, f"{a}/{b}={round(a//b,0)}"])
    return possible


class CountDown(object):
    def __init__(self, max_target=25, start_size=3, min_target=10):
        self.max_target = max_target
        self.min_target = min_target
        self.start_size = start_size
    
    def generate(self, target):
        if target > self.max_target:
            raise ValueError("Target cannot be greater than max target")
        if target < self.min_target:
            raise ValueError("Target cannot be less than min target")
        
        found = False
        while not found:
            # nums in question can go up to max target
            nums = [random.randint(1, self.max_target-1) for _ in range(self.start_size)]
            solution = self.search(target, nums)
            if solution is not None:
                found = True
        return nums, solution
    
    def get_task(self) -> Dict[str, str]:
        target = random.randint(self.min_target, self.max_target)
        nums, solution = self.generate(target)
        
        self.current_task = {
            'input': f"Find a sequence of arithmetic operations (+, -, *, /) that results in {target} using the numbers {', '.join(map(str, nums))}. Use each number exactly once.",
            'target': target
        }
        return self.current_task

    @staticmethod
    def verify_answer(target: int, nums: str, answer: str) -> bool:
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

    def search(self, target, nums, operations=[]):
        # Navigate the entire solution tree, implemented with DFS
        if len(nums) == 1:
            if nums[0] == target:
                return operations
            else:
                return None

        for i, j in itertools.combinations(range(len(nums)), 2):
            num1, num2 = nums[i], nums[j]
            remaining_nums = [nums[k] for k in range(len(nums)) if k != i and k != j]
            for result, operation in combine_nums(num1, num2):
                new_nums = remaining_nums + [result]
                new_operations = operations + [operation]
                solution = self.search(target, new_nums, new_operations)
                if solution is not None:
                    return solution
        return None

def create_countdown_datasets(
    seed=42,
    num_samples=100000,
):
    random.seed(seed)
    countdown = CountDown(start_size=3, max_target=25, min_target=10)

    train_data = []
    val_data = []
    test_data = []

    for _ in range(num_samples):
        task = countdown.get_task()
        train_data.append((task['input'], task['target']))

    eval_size = 500 
    for _ in range(eval_size):
        task = countdown.get_task()
        val_data.append((task['input'], task['target']))
        task = countdown.get_task()    
        test_data.append((task['input'], task['target']))

    return train_data, val_data, test_data

if __name__ == "__main__":
#     countdown = CountDown()
#     task = countdown.get_task()
#     print(task)
#     # get answer
#     answer = """
# Step 1: 1+2=3
# Step 2: 3*3=9
# Step 3: 9*3=27
# Step 4: 27+3=30
# """
#     q="Find a sequence of arithmetic operations (+, -, *, /) that results in 14 using the numbers 2, 24, 12"
#     answer = """
#  Step 1: 24/2 = 12
#  Step 2: 12 + 2 = 14
# """
#     print(countdown.verify_answer(14, q, answer))
    train_data, val_data, test_data = create_countdown_datasets()
    print(len(train_data), len(val_data), len(test_data))
    # save to file
    data = {}
    data['train'] = train_data
    data['validation'] = val_data
    data['test'] = test_data
    import json
    with open('./countdown.json', 'w') as f:
        json.dump(data, f, indent=4)

