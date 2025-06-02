import os, re, json, pandas as pd
from google import genai
from dotenv import load_dotenv
import time
load_dotenv()

_PROMPT = (
    "this is footage from a water polo game. i want you to draw a box around the main action of the game (where the majority of the players are, and if a team is scoring or on offense, the box should include the goal they want to score on). the purpose of this box is to determine the optimal zoom in frame for the camerman, as in many frames, the camera should have been more zoomed in. go at 1 fps and go in intervals of three seconds. you will be returning data about the timestamp and the coordinates of the box (top left x, top left y, bottom right x, bottom right y)."
    "Reply as JSON list of objects {{t_ms:int,x1:int,y1:int,x2:int,y2:int}}."
)
print("running")
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))  
def upload_and_prompt(mp4_path: str) -> pd.DataFrame:
    print("uploading file")
    file_ref = client.files.upload(file=mp4_path)
    print(f"file uploaded: {file_ref.name}, state: {file_ref.state}")

    # Wait for the file to be active
    # Files are ACTIVE once they have been processed.
    # You can use FileUploader.get_file and check the state of the file.
    while file_ref.state.name != "ACTIVE":
        print(f"File {file_ref.name} is not active yet. Current state: {file_ref.state.name}")
        time.sleep(5)  # Wait for 5 seconds before checking again
        file_ref = client.files.get(name=file_ref.name)
        print(f"Re-checked file state: {file_ref.state.name}")

    print(f"File {file_ref.name} is now ACTIVE.")

    # Now that the file is active, you can use it in generate_content
    print("Generating content with the file...")
    resp = client.models.generate_content(
        model="gemini-2.5-flash-preview-05-20", 
        contents=[file_ref, _PROMPT]
    )
    print("response received")
    # Gemini may wrap JSON in text; find the first {...]
    boxes_json_match = re.search(r"\[.*\]", resp.text, re.S)
    if not boxes_json_match:
        print("Error: Could not find JSON in the response.")
        print("Full response text:", resp.text)
        return pd.DataFrame() # Return empty DataFrame or raise an error
        
    boxes_json = boxes_json_match.group()
    records = json.loads(boxes_json)
    print("records loaded")
    return pd.DataFrame(records)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    video_file_path = os.path.join(project_root, "videos", "waterpolo_trimmed.webm")
    print("video file path:", video_file_path)
    df = upload_and_prompt(video_file_path)
    print(df)