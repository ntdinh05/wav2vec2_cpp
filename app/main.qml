import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

ApplicationWindow {
    visible: true
    width: 700
    height: 500
    title: "Audio Inference Runner"
    color: "#1e1e1e" // Dark background like your screenshot

    signal startProgramClicked()

    // --- LOGIC ---
    function appendToLog(text) {
        // System logs (like "Listening...") go here
        if (text.trim() !== "") {
             logDisplay.text += "> " + text + "\n"
        }
    }

    function updateTranscript(text) {
        // The spoken text is appended CLEANLY to the main terminal area
        // We add a space to separate words
        transcriptDisplay.text += text + " "
    }

    function processFinished() {
        startBtn.enabled = true
        startBtn.text = "Start Listening"
    }

    // --- UI LAYOUT ---
    ColumnLayout {
        anchors.fill: parent
        anchors.margins: 20
        spacing: 15

        // Header Title
        Text {
            text: "Model Inference Output"
            color: "#ffffff"
            font.pixelSize: 18
            font.bold: true
            font.family: "Arial"
            Layout.alignment: Qt.AlignLeft
        }

        // The "Terminal" Box
        Rectangle {
            Layout.fillWidth: true
            Layout.fillHeight: true
            color: "#000000" // Black terminal background
            border.color: "#333333"
            radius: 8

            ScrollView {
                anchors.fill: parent
                anchors.margins: 15
                clip: true

                ColumnLayout {
                    width: parent.width
                    spacing: 10

                    // 1. System Log (Dimmed)
                    Text {
                        id: logDisplay
                        Layout.fillWidth: true
                        color: "#666666" // Dimmed grey for system messages
                        font.family: "Courier New"
                        font.pixelSize: 14
                        wrapMode: Text.WordWrap
                        text: "" 
                    }

                    // 2. Active Transcript (Bright Green)
                    Text {
                        id: transcriptDisplay
                        Layout.fillWidth: true
                        color: "#00FF00" // Hacker Green
                        font.family: "Courier New"
                        font.pixelSize: 16
                        font.bold: true
                        wrapMode: Text.WordWrap
                        text: "" // Starts empty
                    }
                }
            }
        }

        // Start Button (Styled to match dark theme)
        Button {
            id: startBtn
            text: "Start Listening"
            Layout.fillWidth: true
            Layout.preferredHeight: 40
            
            background: Rectangle {
                color: startBtn.down ? "#333333" : "#252526"
                border.color: "#444444"
                radius: 4
            }
            contentItem: Text {
                text: startBtn.text
                color: "#ffffff"
                horizontalAlignment: Text.AlignHCenter
                verticalAlignment: Text.AlignVCenter
            }

            onClicked: {
                startBtn.enabled = false
                startBtn.text = "System Live..."
                transcriptDisplay.text = "" // Clear old text
                logDisplay.text = ""
                startProgramClicked()
            }
        }
    }
}