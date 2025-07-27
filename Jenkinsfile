pipeline {
    agent any

    environment {
        EC2_IP = 'your-ec2-ip'
        EC2_USER = 'ec2-user'
        SSH_KEY = credentials('ec2-ssh-key-id') // from Jenkins credentials
        IMAGE_NAME = 'resume-classifier-api'
    }

    stages {
        stage('Clone Repository') {
            steps {
                git 'https://github.com/krishnadeep0411/-resume-classifier-api.git'
            }
        }

        stage('Build Docker Image') {
            steps {
                sh "docker build -t ${IMAGE_NAME}:latest ."
            }
        }

        stage('Deploy to EC2') {
            steps {
                sshagent (credentials: ['ec2-ssh-key-id']) {
                    sh """
                    ssh -o StrictHostKeyChecking=no ${EC2_USER}@${EC2_IP} '
                        docker stop ${IMAGE_NAME} || true
                        docker rm ${IMAGE_NAME} || true
                        docker rmi ${IMAGE_NAME}:latest || true
                        git clone https://github.com/krishnadeep0411/-resume-classifier-api.git || true
                        cd -resume-classifier-api
                        docker build -t ${IMAGE_NAME}:latest .
                        docker run -d -p 8000:8000 --name ${IMAGE_NAME} ${IMAGE_NAME}:latest
                    '
                    """
                }
            }
        }
    }

    post {
        success {
            echo "Deployed ${IMAGE_NAME} to EC2 successfully!"
        }
        failure {
            echo "Deployment failed."
        }
    }
}
