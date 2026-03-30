/**
 * Shared Jenkins pipeline utilities for vEcoli
 */

def run(Map config) {
    def buildResult = 'SUCCESS'
    
    try {
        stage(config.stageName) {
            def commandJoinSeparator = ' && \n                '
            def commands = config.commands instanceof List ? config.commands.join(commandJoinSeparator) : config.commands
            sh """
            TIMEOUT='${config.timeout}'
            code=0
            timeout \$TIMEOUT ${commands} || code=\$?
            [ \$code -eq 124 ] && echo "Jenkins job timeout after \$TIMEOUT"
            exit \$code
            """
        }
        currentBuild.result = 'SUCCESS'
    } catch (org.jenkinsci.plugins.workflow.steps.FlowInterruptedException e) {
        buildResult = 'ABORTED'
        currentBuild.result = 'ABORTED'
        throw e
    } catch (Exception e) {
        buildResult = 'FAILURE'
        currentBuild.result = 'FAILURE'
        throw e
    } finally {
        // Send notifications based on result
        sendNotification(buildResult)
        
        // Cleanup old outputs
        cleanWs()
        sh """
        target="${config.outputDir}"
        keep=${config.keepCount ?: 20}
        mkdir -p "\$target"
        
        # Safely delete old directories, handling special characters in filenames
        find "\$target" -mindepth 1 -maxdepth 1 -type d -printf '%T@\\t%p\\n' \\
            | sort -rn \\
            | tail -n +\$((keep + 1)) \\
            | cut -f2- \\
            | while IFS= read -r dir; do
                echo "Pruning old directory: \$dir"
                rm -rf -- "\$dir"
            done
        """
    }
}

def sendNotification(String result) {
    def statusChanged = hasStatusChanged()
    
    switch(result) {
        case 'SUCCESS':
            if (statusChanged) {
                slackSend(
                    color: 'good',
                    channel: '#jenkins',
                    message: "${env.JOB_NAME} ${env.BUILD_DISPLAY_NAME} back to normal after ${currentBuild.durationString.minus(' and counting')} (<${env.BUILD_URL}|link>)"
                )
            } else {
                echo "Job was successful just like last run, not sending Slack notification."
            }
            break
            
        case 'ABORTED':
            def abortedMessage = statusChanged ? 
                "${env.JOB_NAME} ${env.BUILD_DISPLAY_NAME} aborted after ${currentBuild.durationString.minus(' and counting')} (<${env.BUILD_URL}|link>)" :
                "${env.JOB_NAME} ${env.BUILD_DISPLAY_NAME} aborted again after ${currentBuild.durationString.minus(' and counting')} (<${env.BUILD_URL}|link>)"
            slackSend(
                color: '#808080',
                channel: '#jenkins',
                message: abortedMessage
            )
            break
            
        case 'FAILURE':
            if (statusChanged) {
                slackSend(
                    color: 'danger',
                    channel: '#jenkins',
                    message: "${env.JOB_NAME} ${env.BUILD_DISPLAY_NAME} failed after ${currentBuild.durationString.minus(' and counting')} (<${env.BUILD_URL}|link>)"
                )
            } else {
                slackSend(
                    color: 'danger',
                    channel: '#jenkins',
                    message: "${env.JOB_NAME} ${env.BUILD_DISPLAY_NAME} is still failing after ${currentBuild.durationString.minus(' and counting')} (<${env.BUILD_URL}|link>)"
                )
            }
            break
    }
}

def hasStatusChanged() {
    if (currentBuild.previousBuild == null) {
        return true
    }
    
    def previousResult = currentBuild.previousBuild.result
    def currentResult = currentBuild.result
    
    return previousResult != currentResult
}

return this
