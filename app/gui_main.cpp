#include <QGuiApplication>
#include <QQmlApplicationEngine>
#include <QProcess>
#include <QObject>
#include <QQmlContext>
#include <QDebug>
#include <QDir>
#include <QMetaObject>

// --- CONFIGURATION ---
const QString INFERENCE_EXE_NAME = "auto_resample";

class Backend : public QObject {
    Q_OBJECT

public:
    explicit Backend(QObject *parent = nullptr) : QObject(parent), m_process(nullptr) {}
    Q_INVOKABLE void handleStart() {
        if (m_process) {
            m_process->kill();
            delete m_process;
        }
        m_process = new QProcess(this);
        m_process->setProcessChannelMode(QProcess::MergedChannels);

        // Connect signals
        connect(m_process, &QProcess::readyReadStandardOutput, this, &Backend::readOutput);
        connect(m_process, QOverload<int, QProcess::ExitStatus>::of(&QProcess::finished), 
                this, &Backend::processFinished);

        // --- FIND THE EXECUTABLE ---
        QString program = QCoreApplication::applicationDirPath() + "/" + INFERENCE_EXE_NAME;
        if (!QFile::exists(program)) {
            // Check if it's in a sibling 'build' dir relative to source
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
    // Signal to notify QML
    void finished();

private slots:
    void readOutput() {
        while (m_process->canReadLine()) {
            QString line = QString::fromUtf8(m_process->readLine()).trimmed();
            // --- PARSE TAGS ---
            if (line.startsWith("TR:")) {
                QString content = line.mid(3).trimmed();
                updateTranscript(content);
            } else {
                sendToLog(line);
            }
        }
    }

    void processFinished(int exitCode, QProcess::ExitStatus exitStatus) {
        Q_UNUSED(exitCode);
        Q_UNUSED(exitStatus);
        sendToLog("Process ended.");
        QQmlApplicationEngine* engine = qobject_cast<QQmlApplicationEngine*>(parent());
        if (engine && !engine->rootObjects().isEmpty()) {
             QMetaObject::invokeMethod(engine->rootObjects().first(), "processFinished");
        }
    }

private:
    QProcess *m_process;
    void sendToLog(const QString &text) {
        QQmlApplicationEngine* engine = qobject_cast<QQmlApplicationEngine*>(parent());
        if (!engine || engine->rootObjects().isEmpty()) return;
        QObject *root = engine->rootObjects().first();
        QMetaObject::invokeMethod(root, "appendToLog", Q_ARG(QVariant, text));
    }
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
    
    Backend *backend = new Backend(&engine);

    // Load the QML
    const QUrl url(QStringLiteral("qrc:/main.qml"));
    engine.load(QUrl::fromLocalFile("main.qml"));

    if (engine.rootObjects().isEmpty())
        return -1;

    QObject *root = engine.rootObjects().first();
    QObject::connect(root, SIGNAL(startProgramClicked()), backend, SLOT(handleStart()));

    return app.exec();

}
#include "gui_main.moc"