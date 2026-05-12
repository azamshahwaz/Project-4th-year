import os

# =====================================================
# LOG DIRECTORY
# =====================================================

LOG_DIR = "logs"

if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# =====================================================
# GLOBAL LOG FILE
# =====================================================

LOG_FILE = None


# =====================================================
# SET LOG FILE
# =====================================================

def set_log_file(dataset_name):

    global LOG_FILE

    dataset_name = os.path.splitext(dataset_name)[0]

    safe_name = dataset_name.replace(" ", "_")

    LOG_FILE = os.path.join(
        LOG_DIR,
        f"{safe_name}_report.txt"
    )

    # CLEAR OLD CONTENT
    open(LOG_FILE, "w").close()


# =====================================================
# WRITE LOG
# =====================================================

def write_log(message):

    global LOG_FILE

    with open(LOG_FILE, "a", encoding="utf-8") as f:

        f.write(str(message) + "\n")


# =====================================================
# LOG SECTION
# =====================================================

def log_section(title):

    global LOG_FILE

    separator = "=" * 60

    with open(LOG_FILE, "a", encoding="utf-8") as f:

        f.write("\n")
        f.write(separator + "\n")
        f.write(title + "\n")
        f.write(separator + "\n")