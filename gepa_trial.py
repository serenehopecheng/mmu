import os, json, random, re
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Configuration
MODEL = "gpt-5-nano"
ITERATIONS = 5
TASKS = {
    "3.mp4": "A person correctly using a French Press to brew coffee, countertop view.",
    "4.mp4": "A close-up of a Newton's Cradle in motion on a desk, macro view.",
    "5.mp4": "A person tying a Bowline knot with a thick rope, hands-only close-up.",
}

SEED_PROMPT = """Write an 8-second video prompt for Veo 3. 
Rules: 
1. Output valid JSON: {"script": "...", "narration": "..."}.
2. One continuous shot, no cuts. 
3. Specify camera angle and lighting.
4. Narration must be under 18 words and provide context not seen on screen.
Topic: {topic}"""

def call_llm(prompt, json_mode=True):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"} if json_mode else None
    )
    return response.choices[0].message.content

def evaluate(system_prompt, topic):
    # 1. Generate the video script
    generated = call_llm(system_prompt.replace("{topic}", topic))
    
    # 2. Score the generated script
    eval_query = f"""Evaluate this Veo 3 prompt based on the retrieval of information necessary to include in the video for the topic: {topic}.
    {generated}
    
    Return JSON: {{"score": 0-10, "feedback": "one sentence of what to fix"}}"""
    eval_res = json.loads(call_llm(eval_query))
    return eval_res.get("score", 0), eval_res.get("feedback", ""), generated

def mutate(parent_prompt, feedback_list):
    feedback_str = "\n".join(feedback_list)
    mutate_query = f"""Improve this system prompt based on feedback.
    Original: {parent_prompt}
    Feedback: {feedback_str}
    
    Return the full improved prompt text only."""
    return call_llm(mutate_query, json_mode=False)

def run_evolution():
    current_prompt = SEED_PROMPT
    best_score = 0
    run_data = []
    
    print(f"Starting evolution for {ITERATIONS} iterations...")

    for i in range(ITERATIONS):
        scores = []
        feedbacks = []
        generated_scripts = []
        
        # Test against all tasks
        for topic in TASKS.values():
            score, feedback, generated = evaluate(current_prompt, topic)
            scores.append(score)
            feedbacks.append(f"- {topic}: {feedback}")
            generated_scripts.append({topic: generated})
        
        avg_score = sum(scores) / len(scores)
        print(f"Iteration {i}: Avg Score = {avg_score:.2f}")

        # Collect data for this iteration
        iteration_data = {
            "iteration": i,
            "prompt": current_prompt,
            "scores": dict(zip(TASKS.keys(), scores)),
            "avg_score": avg_score,
            "feedbacks": feedbacks,
            "generated_scripts": generated_scripts
        }
        run_data.append(iteration_data)

        if avg_score > best_score:
            best_score = avg_score
            with open("best_prompt.txt", "w") as f:
                f.write(current_prompt)
        
        # Improve the prompt for the next round
        current_prompt = mutate(current_prompt, feedbacks)

    # Save run data to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_file = f"runs/gepa_trial_{timestamp}.json"
    with open(run_file, "w") as f:
        json.dump(run_data, f, indent=2)
    
    print("-" * 30)
    print(f"Evolution complete. Best Score: {best_score}")
    print("Final prompt saved to best_prompt.txt")
    print(f"Run data saved to {run_file}")

if __name__ == "__main__":
    run_evolution()