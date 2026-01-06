import sys
import os
import subprocess
from PySide6.QtGui import QGuiApplication
from PySide6.QtQml import QQmlApplicationEngine
from PySide6.QtCore import QObject, Slot, Signal, QThread

# --- CONFIGURATION ---
# Ensure this matches your C++ executable name
EXECUTABLE_NAME = "auto_resample" 

class ProcessRunner(QThread):
    log_received = Signal(str)        # For system messages (Ready, Loading...)
    transcript_received = Signal(str) # For the CLEAN spoken text
    finished_signal = Signal()

    def run(self):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        build_dir = os.path.abspath(os.path.join(script_dir, "..", "build"))
        exe_path = os.path.join(build_dir, EXECUTABLE_NAME)

        if not os.path.exists(exe_path):
            self.log_received.emit(f"Error: Executable not found at:\n{exe_path}")
            self.finished_signal.emit()
            return

        try:
            process = subprocess.Popen(
                [exe_path],
                cwd=build_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1
            )

            # Read output line by line
            while True:
                line = process.stdout.readline()
                if not line and process.poll() is not None:
                    break
                if line:
                    clean_line = line.strip()
                    # DETECT THE TAG FROM C++
                    if clean_line.startswith("TR:"):
                        # It's spoken text: Remove "TR:" and update the main display
                        content = clean_line[3:].strip()
                        self.transcript_received.emit(content)
                    else:
                        # It's a system log: Print to the console log
                        self.log_received.emit(clean_line)

            process.wait()
            
        except Exception as e:
            self.log_received.emit(f"Error: {str(e)}")

        self.finished_signal.emit()

class Backend(QObject):
    def __init__(self, engine):
        super().__init__()
        self.engine = engine
        self.runner = None

    @Slot()
    def handle_start(self):
        self.runner = ProcessRunner()
        self.runner.log_received.connect(self.send_log)
        self.runner.transcript_received.connect(self.update_transcript)
        self.runner.finished_signal.connect(self.enable_ui)
        self.runner.start()

    def send_log(self, text):
        root = self.engine.rootObjects()[0]
        # Only show logs if they aren't empty
        if text:
            root.appendToLog(text)

    def update_transcript(self, text):
        root = self.engine.rootObjects()[0]
        root.updateTranscript(text)

    def enable_ui(self):
        root = self.engine.rootObjects()[0]
        root.processFinished()

if __name__ == "__main__":
    app = QGuiApplication(sys.argv)
    engine = QQmlApplicationEngine()

    engine.load(os.path.join(os.path.dirname(__file__), "main.qml"))

    if not engine.rootObjects():
        sys.exit(-1)

    backend = Backend(engine)
    root = engine.rootObjects()[0]
    # Auto-connect start signal
    root.startProgramClicked.connect(backend.handle_start)

    sys.exit(app.exec())