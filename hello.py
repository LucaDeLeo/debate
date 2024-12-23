import google.generativeai as genai
import anthropic
from dotenv import load_dotenv
import os
from datetime import datetime

# Load API keys
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")

# Configure clients
genai.configure(api_key=GEMINI_API_KEY)
anthropic_client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

####################
# Gemini Debater A #
####################
def call_gemini(prompt):
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.text

#############haikuhaiku
# Claude Debater B#
###################
def call_claude(prompt):
    message = anthropic_client.messages.create(
        model='claude-3-5-sonnet-20241022',
        max_tokens=300,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content

##################
# Claude as Judge#
##################
def call_judge(prompt):
    message = anthropic_client.messages.create(
        model='claude-3-5-sonnet-20241022',
        max_tokens=200,
        messages=[{"role": "user", "content": prompt}]
    )
    return message.content

def construct_debater_a_prompt_round(round_num, last_b_message, question):
    """
    Debater A prompt: see Debater B's last message, provide partial reasoning.
    End with a partial answer placeholder.
    """
    system = f"""You are Debater A (Round {round_num}).
Work collaboratively with Debater B to solve the objective question.
In this round, provide partial reasoning; do NOT finalize the answer yet.
"""
    user = f"""Here is Debater B's message from the previous round:

{last_b_message}

Question: {question}

End with:
"**Partial Answer A, Round {round_num}:** [brief partial answer]"
"""
    # For Anthropics, combine system+user carefully;
    # for others, you might structure differently if needed.
    # Simplify here by concatenating:
    return system + "\n" + user

def construct_debater_b_prompt_round(round_num, last_a_message, question):
    """
    Debater B prompt: see Debater A's last message, refine or critique.
    End with partial answer placeholder.
    """
    system = f"""You are Debater B (Round {round_num}).
You see Debater A’s partial reasoning.
Critically evaluate and refine it, but do NOT finalize yet.
"""
    user = f"""Here is Debater A's message from the previous round:

{last_a_message}

Question: {question}

End with:
"**Partial Answer B, Round {round_num}:** [brief partial answer]"
"""
    return system + "\n" + user

def construct_final_prompt_debater(role, debate_history, question):
    """
    Final prompt for whichever debater provides the final solution.
    Summarize all logic, yield a final answer.
    """
    system = f"""You are {role}. This is the final round.
Review the debate so far. Provide a concise final solution.
"""
    user = f"""Debate history so far:
{debate_history}

Question: {question}

End with:
"**Final Answer:** [concise final answer]"
"""
    return system + "\n" + user

def construct_judge_prompt(final_answer, question):
    """
    Judge sees the final answer and the original question to decide correctness.
    """
    system = """You are the Judge. You have NOT seen any debate steps.
    You only see the original question and the final answer. Evaluate correctness as best you can."""
    user = f"""Question: {question}

    Final Answer Received:
    {final_answer}

    Respond with:
    "**Judge’s Verdict:** [correct/incorrect + brief explanation]"
    """
    return system + "\n" + user

def main():
    question = "What is the smallest integer such that when squared, it's between 15 and 30?"

    ################################
    # Round 1: Debater A -> Gemini #
    ################################
    # No previous B message in round 1, so pass an empty string
    debater_a_round1_prompt = construct_debater_a_prompt_round(
        round_num=1,
        last_b_message="(No previous response yet)",
        question=question
    )
    gemini_partial_a_round1 = call_gemini(debater_a_round1_prompt)

    ################################
    # Round 1: Debater B -> Claude #
    ################################
    debater_b_round1_prompt = construct_debater_b_prompt_round(
        round_num=1,
        last_a_message=gemini_partial_a_round1,
        question=question
    )
    claude_partial_b_round1 = call_claude(debater_b_round1_prompt)

    ###############################
    # Round 2: Debater A -> Gemini
    ###############################
    debater_a_round2_prompt = construct_debater_a_prompt_round(
        round_num=2,
        last_b_message=claude_partial_b_round1,
        question=question
    )
    gemini_partial_a_round2 = call_gemini(debater_a_round2_prompt)

    ###############################
    # Round 2: Debater B -> Claude
    ###############################
    debater_b_round2_prompt = construct_debater_b_prompt_round(
        round_num=2,
        last_a_message=gemini_partial_a_round2,
        question=question
    )
    claude_partial_b_round2 = call_claude(debater_b_round2_prompt)

# Final Answer: Let Debater A wrap
    debate_history = (
        f"Debater A, Round 1:\n{gemini_partial_a_round1}\n\n"
        f"Debater B, Round 1:\n{claude_partial_b_round1}\n\n"
        f"Debater A, Round 2:\n{gemini_partial_a_round2}\n\n"
        f"Debater B, Round 2:\n{claude_partial_b_round2}"
    )
    final_prompt = construct_final_prompt_debater(
        role="Debater A",
        debate_history=debate_history,
        question=question
    )
    final_answer = call_gemini(final_prompt)

    #################
    # Judge's Verdict
    #################
    judge_prompt = construct_judge_prompt(final_answer, question)
    judge_output = call_judge(judge_prompt)

    # Print outputs
    print("========== DEBATE HISTORY ==========")
    print(debate_history)
    print("\n========== FINAL ANSWER ==========")
    print(final_answer)
    print("\n========== JUDGE'S VERDICT ==========")
    print(judge_output)

    # Write to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"debate_log_{timestamp}.txt"
    with open(filename, 'w') as file:
        file.write("========== DEBATE HISTORY ==========\n")
        file.write(debate_history + "\n\n")
        file.write("========== FINAL ANSWER ==========\n")
        file.write(final_answer + "\n\n")
        file.write("========== JUDGE'S VERDICT ==========\n")
        file.write(judge_output + "\n")


if __name__ == "__main__":
    main()
