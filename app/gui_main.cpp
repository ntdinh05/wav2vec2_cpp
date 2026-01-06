#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QProcess>
#include <QObject>
#include <QQmlContext>
#include <QDebug>
#include <QDir>
#include <QMetaObject>

// --- CONFIGURATION ---
// Ensure this matches the name of your OTHER compiled executable
const QString INFERENCE_EXE_NAME = "auto_resample";

class Backend : public QObject {
    Q_OBJECT

public:
    explicit Backend(QObject *parent = nullptr) : QObject(parent), m_process(nullptr) {}

    // Allow QML to call this function
    Q_INVOKABLE void handleStart() {
        if (m_process) {
            m_process->kill();
            delete m_process;
        }

        m_process = new QProcess(this);
        
        // Merge Stderr into Stdout so we catch everything
        m_process->setProcessChannelMode(QProcess::MergedChannels);

        // Connect signals
        connect(m_process, &QProcess::readyReadStandardOutput, this, &Backend::readOutput);
        connect(m_process, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), 
                this, &Backend::processFinished);

        // --- FIND THE EXECUTABLE ---
        // We assume the build folder layout is:
        // /build/InferenceRunner (This App)
        // /build/auto_resample   (The Model App)
        QString program = QCoreApplication::applicationDirPath() + "/" + INFERENCE_EXE_NAME;
        
        // Use QDir::cleanPath to resolve "../" if necessary
        // If your layout is different, adjust this path logic.
        if (!QFile::exists(program)) {
            // Fallback: Check if it's in a sibling 'build' dir relative to source
            program = QCoreApplication::applicationDirPath() + "/../build/" + INFERENCE_EXE_NAME;
        }

        if (!QFile::exists(program)) {
            sendToLog("Error: Could not find executable at:\n" + program);
            emit finished(); // Re-enable button
            return;
        }

        m_process->start(program);
        if (!m_process->waitForStarted()) {
            sendToLog("Error: Failed to start process.");
            emit finished();
        }
    }

signals:
    // Signal to notify QML that we are done
    void finished();

private slots:
    void readOutput() {
        while (m_process->canReadLine()) {
            QString line = QString::fromUtf8(m_process->readLine()).trimmed();
            
            // --- PARSE TAGS ---
            if (line.startsWith("TR:")) {
                // It's a Transcript: Remove "TR:" and update Main Display
                QString content = line.mid(3).trimmed();
                updateTranscript(content);
            } else {
                // It's a Log: Send to bottom text area
                sendToLog(line);
            }
        }
    }

    void processFinished(int exitCode, QProcess::ExitStatus exitStatus) {
        Q_UNUSED(exitCode);
        Q_UNUSED(exitStatus);
        sendToLog("Process ended.");
        
        // Find QML root and call processFinished()
        // We do this via signal/slot or invokeMethod
        QQmlApplicationEngine* engine = qobject_cast<QQmlApplicationEngine*>(parent());
        if (engine && !engine->rootObjects().isEmpty()) {
             QMetaObject::invokeMethod(engine->rootObjects().first(), "processFinished");
        }
    }

private:
    QProcess *m_process;

    // Helper to call QML function "appendToLog"
    void sendToLog(const QString &text) {
        QQmlApplicationEngine* engine = qobject_cast<QQmlApplicationEngine*>(parent());
        if (!engine || engine->rootObjects().isEmpty()) return;

        QObject *root = engine->rootObjects().first();
        QMetaObject::invokeMethod(root, "appendToLog", Q_ARG(QVariant, text));
    }

    // Helper to call QML function "updateTranscript"
    void updateTranscript(const QString &text) {
        QQmlApplicationEngine* engine = qobject_cast<QQmlApplicationEngine*>(parent());
        if (!engine || engine->rootObjects().isEmpty()) return;

        QObject *root = engine->rootObjects().first();
        QMetaObject::invokeMethod(root, "updateTranscript", Q_ARG(QVariant, text));
    }
};

int main(int argc, char *argv[])
{
    QGuiApplication app(argc, argv);

    QQmlApplicationEngine engine;
    
    // Create our backend and set the engine as parent so it can find rootObjects
    Backend *backend = new Backend(&engine);

    // Load the QML
    const QUrl url(QStringLiteral("qrc:/main.qml"));
    // Since we are loading from file system for simplicity in this example:
    engine.load(QUrl::fromLocalFile("main.qml"));

    if (engine.rootObjects().isEmpty())
        return -1;

    // Connect the QML signal "startProgramClicked" to our C++ slot
    QObject *root = engine.rootObjects().first();
    QObject::connect(root, SIGNAL(startProgramClicked()), backend, SLOT(handleStart()));

    return app.exec();

}
#include "gui_main.moc"