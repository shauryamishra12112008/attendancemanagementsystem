
import json
import os
import re
import tempfile
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import InferenceClient


# --------------------------------------------------
# 1. CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "attendance.json"
ENV_FILE = BASE_DIR / ".env"

load_dotenv(dotenv_path=ENV_FILE, override=True)

MODEL = os.getenv("HF_MODEL", "").strip()
TOKEN = os.getenv("HF_TOKEN", "").strip()

client = None

if MODEL and TOKEN:
    client = InferenceClient(
        model=MODEL,
        token=TOKEN,
        provider="auto"
    )


SYSTEM_PROMPT = """
You are an attendance management assistant.

Convert the user's command into exactly one JSON object.
Return JSON only. Do not use Markdown.

Supported actions:

1. {"action":"student_attendance","roll_no":101}
2. {"action":"student_attendance","name":"Rahul Sharma"}
3. {"action":"average_attendance"}
4. {"action":"list_students"}
5. {"action":"below_threshold","threshold":75}
6. {"action":"mark_attendance","roll_no":101,
    "day":"Day 1","status":"present"}
7. {"action":"remove_attendance","roll_no":101,
    "day":"Day 1"}
8. {"action":"clarify","message":"Ask for missing information"}

Rules:
- roll_no must be a student's roll number.
- day must be in the format "Day 1", "Day 2", etc.
- status must be "present" or "absent".
- Never invent a roll number, student, or day.
- If required information is missing, return clarify.
"""


# --------------------------------------------------
# 2. LOAD AND SAVE ATTENDANCE DATA
# --------------------------------------------------

def load_data():
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"attendance.json was not found at: {DATA_FILE}"
        )

    with open(DATA_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict) or not isinstance(
        data.get("students"), list
    ):
        raise ValueError(
            "attendance.json must contain a 'students' list."
        )

    for student in data["students"]:
        if not isinstance(student, dict):
            raise ValueError("Invalid student record in attendance.json.")

        if "name" not in student or "roll_no" not in student:
            raise ValueError(
                "Every student must have name and roll_no fields."
            )

        if not isinstance(student.get("attendance"), dict):
            raise ValueError(
                f"Attendance data is missing for {student['name']}."
            )

        for day, status in student["attendance"].items():
            if status not in ("present", "absent", None):
                raise ValueError(
                    f"Invalid status for {student['name']}, {day}. "
                    "Use present, absent, or null."
                )

    return data


def save_data(data):
    """Save safely by writing a temporary file first."""
    fd, temp_path = tempfile.mkstemp(
        dir=str(BASE_DIR),
        suffix=".tmp",
        prefix="attendance_"
    )

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4, ensure_ascii=False)
            file.write("\n")

        os.replace(temp_path, DATA_FILE)

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


# --------------------------------------------------
# 3. FIND A STUDENT
# --------------------------------------------------

def find_student(data, roll_no=None, name=None):
    students = data["students"]

    if roll_no is not None:
        for student in students:
            if str(student["roll_no"]) == str(roll_no):
                return student
        return None

    if name:
        name = name.strip().lower()

        # Prefer an exact name match.
        for student in students:
            if student["name"].strip().lower() == name:
                return student

        # Allow a unique partial name match.
        matches = [
            student for student in students
            if name in student["name"].strip().lower()
        ]

        if len(matches) == 1:
            return matches[0]

        if len(matches) > 1:
            raise ValueError(
                "More than one student matches that name. "
                "Please provide the full name or roll number."
            )

    return None


# --------------------------------------------------
# 4. ATTENDANCE CALCULATIONS
# --------------------------------------------------

def attendance_stats(student):
    attendance = student["attendance"]

    recorded = [
        status for status in attendance.values()
        if status in ("present", "absent")
    ]

    total_days = len(recorded)
    days_present = recorded.count("present")
    days_absent = recorded.count("absent")

    percentage = (
        round((days_present / total_days) * 100, 2)
        if total_days else 0.0
    )

    return {
        "name": student["name"],
        "roll_no": student["roll_no"],
        "total_days": total_days,
        "days_present": days_present,
        "days_absent": days_absent,
        "attendance_percentage": percentage
    }


# --------------------------------------------------
# 5. EXECUTE AN ACTION
# --------------------------------------------------

def execute_tool(data, request):
    action = request.get("action")

    if action == "clarify":
        return {
            "message": request.get(
                "message", "Please provide more information."
            )
        }

    if action == "list_students":
        return {
            "students": [
                {
                    "name": student["name"],
                    "roll_no": student["roll_no"]
                }
                for student in data["students"]
            ]
        }

    if action == "average_attendance":
        results = [
            attendance_stats(student)
            for student in data["students"]
        ]

        average = (
            round(
                sum(item["attendance_percentage"] for item in results)
                / len(results),
                2
            )
            if results else 0.0
        )

        return {
            "average_attendance_percentage": average,
            "student_count": len(results),
            "student_details": results
        }

    if action == "below_threshold":
        try:
            threshold = float(request.get("threshold", 75))
        except (TypeError, ValueError):
            raise ValueError("The threshold must be a number.")

        if not 0 <= threshold <= 100:
            raise ValueError("The threshold must be between 0 and 100.")

        results = [
            attendance_stats(student)
            for student in data["students"]
        ]

        return {
            "threshold": threshold,
            "students_below_threshold": [
                item for item in results
                if item["attendance_percentage"] < threshold
            ]
        }

    if action in ("student_attendance", "mark_attendance",
                  "remove_attendance"):

        roll_no = request.get("roll_no")
        name = request.get("name")

        if roll_no is None and not name:
            raise ValueError(
                "Please provide the student's name or roll number."
            )

        student = find_student(data, roll_no=roll_no, name=name)

        if student is None:
            raise ValueError("Student not found in attendance.json.")

        if action == "student_attendance":
            return attendance_stats(student)

        day = request.get("day")

        if not isinstance(day, str) or not re.fullmatch(
            r"Day\s+\d+", day, re.IGNORECASE
        ):
            raise ValueError(
                "Please specify a day, such as Day 1 or Day 2."
            )

        # Match the day key without depending on capitalization.
        matching_day = next(
            (
                existing_day
                for existing_day in student["attendance"]
                if existing_day.lower() == day.lower()
            ),
            None
        )

        if matching_day is None:
            raise ValueError(
                f"{day} does not exist for this student. "
                "Check the days in attendance.json."
            )

        if action == "mark_attendance":
            status = request.get("status")

            if status not in ("present", "absent"):
                raise ValueError(
                    "Attendance status must be present or absent."
                )

            student["attendance"][matching_day] = status
            save_data(data)

            return {
                "message": "Attendance updated successfully.",
                "student": student["name"],
                "roll_no": student["roll_no"],
                "day": matching_day,
                "status": status,
                "updated_attendance": attendance_stats(student)
            }

        if action == "remove_attendance":
            student["attendance"][matching_day] = None
            save_data(data)

            return {
                "message": "Attendance entry removed successfully.",
                "student": student["name"],
                "roll_no": student["roll_no"],
                "day": matching_day,
                "updated_attendance": attendance_stats(student)
            }

    raise ValueError("Unsupported action requested.")


# --------------------------------------------------
# 6. LOCAL COMMAND RECOGNITION
# --------------------------------------------------

def get_local_request(user_input, data):
    """
    Basic fallback that works without API credits.
    Supports common attendance commands.
    """
    text = user_input.lower().strip()

    # Find a roll number.
    roll_match = re.search(
        r"\broll(?:\s*(?:number|no\.?))?\s*[:#-]?\s*(\d+)\b",
        text
    )

    if not roll_match:
        roll_match = re.search(r"\broll\s*#?\s*(\d+)\b", text)

    roll_no = int(roll_match.group(1)) if roll_match else None

    # Find a day.
    day_match = re.search(r"\bday\s*(\d+)\b", text)
    day = f"Day {day_match.group(1)}" if day_match else None

    # Identify a student by name if no roll number was supplied.
    name = None

    if roll_no is None:
        for student in data["students"]:
            student_name = student["name"].lower()

            if student_name in text:
                name = student["name"]
                break

    # List students.
    if (
        "list students" in text
        or "show students" in text
        or "all students" in text
    ):
        return {"action": "list_students"}

    # Average attendance.
    if "average" in text and "attendance" in text:
        return {"action": "average_attendance"}

    # Attendance below a percentage threshold.
    if "below" in text and "attendance" in text:
        threshold_match = re.search(
            r"(\d+(?:\.\d+)?)\s*%", text
        )

        threshold = (
            float(threshold_match.group(1))
            if threshold_match else 75.0
        )

        return {
            "action": "below_threshold",
            "threshold": threshold
        }

    # Remove an attendance entry.
    if "remove" in text and "attendance" in text:
        if roll_no is None and name is None:
            raise ValueError(
                "Please include the student's name or roll number."
            )

        if day is None:
            raise ValueError("Please specify the day, such as Day 1.")

        request = {
            "action": "remove_attendance",
            "day": day
        }

        if roll_no is not None:
            request["roll_no"] = roll_no
        else:
            request["name"] = name

        return request

    # Mark attendance present or absent.
    if (
        ("mark" in text or "set" in text or "update" in text)
        and ("present" in text or "absent" in text)
    ):
        if roll_no is None and name is None:
            raise ValueError(
                "Please include the student's name or roll number."
            )

        if day is None:
            raise ValueError("Please specify the day, such as Day 1.")

        request = {
            "action": "mark_attendance",
            "day": day,
            "status": "absent" if "absent" in text else "present"
        }

        if roll_no is not None:
            request["roll_no"] = roll_no
        else:
            request["name"] = name

        return request

    # Attendance for an individual student.
    if "attendance" in text and (roll_no is not None or name is not None):
        request = {"action": "student_attendance"}

        if roll_no is not None:
            request["roll_no"] = roll_no
        else:
            request["name"] = name

        return request

    raise ValueError(
        "I could not understand that command without AI. "
        "Try: 'show attendance for roll 101', "
        "'average attendance', or "
        "'mark roll 101 present on Day 1'."
    )


# --------------------------------------------------
# 7. ASK HUGGING FACE AI
# --------------------------------------------------

def get_ai_request(user_input):
    if client is None:
        raise RuntimeError(
            "HF_TOKEN or HF_MODEL is missing from the .env file."
        )

    response = client.chat_completion(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_input}
        ],
        max_tokens=250,
        temperature=0.1
    )

    content = response.choices[0].message.content

    if not content:
        raise ValueError("The AI returned an empty response.")

    # Remove Markdown fences if the model adds them.
    content = content.strip()
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)

    request = json.loads(content)

    if not isinstance(request, dict):
        raise ValueError("The AI response must be a JSON object.")

    allowed_actions = {
        "student_attendance",
        "average_attendance",
        "list_students",
        "below_threshold",
        "mark_attendance",
        "remove_attendance",
        "clarify"
    }

    if request.get("action") not in allowed_actions:
        raise ValueError("The AI returned an unsupported action.")

    return request


# --------------------------------------------------
# 8. MAIN PROGRAM
# --------------------------------------------------

def main():
    print("=" * 48)
    print("       AI ATTENDANCE MANAGEMENT SYSTEM")
    print("=" * 48)

    try:
        data = load_data()
    except (OSError, json.JSONDecodeError, ValueError) as error:
        print(f"Could not load attendance data: {error}")
        return

    if client is None:
        print("AI API is not configured.")
        print("Local command mode will be used.")
    else:
        print(f"Configured AI model: {MODEL}")
        print("AI mode enabled; local fallback is available.")

    print("\nExamples:")
    print("  Show attendance for roll 101")
    print("  What is the average attendance?")
    print("  Show students below 75% attendance")
    print("  Mark roll 101 present on Day 1")
    print("  Remove attendance for roll 102 on Day 3")
    print("  List students")
    print("\nType 'exit' to close the program.")

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nProgram closed.")
            break

        if user_input.lower() in ("exit", "quit", "stop"):
            print("Goodbye!")
            break

        if not user_input:
            continue

        # First try AI. If the API fails or has no credits,
        # use local recognition for supported commands.
        try:
            request = get_ai_request(user_input)
            mode = "AI"
        except Exception as error:
            print(f"\nAI unavailable: {error}")
            print("Trying local command recognition...")
            try:
                request = get_local_request(user_input, data)
                mode = "Local"
            except ValueError as local_error:
                print(f"Could not process command: {local_error}")
                continue

        try:
            # Reload before each action to use the latest saved data.
            data = load_data()
            result = execute_tool(data, request)

            print(f"\n[{mode} mode] Result:")
            print(json.dumps(result, indent=4, ensure_ascii=False))

        except (ValueError, OSError, json.JSONDecodeError) as error:
            print(f"\nError: {error}")


if __name__ == "__main__":
    main()