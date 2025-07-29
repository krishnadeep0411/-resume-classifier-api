pipeline {
    agent any

    environment {
        DOCKER_IMAGE = "krishnadeep0411/resume-classifier-api"
    }

    stages {
        stage('Checkout') {
            steps {
                git 'https://github.com/krishnadeep0411/-resume-classifier-api.git'
            }
        }

        stage('Build Docker Image') {
            steps {
                script {
                    dockerImage = docker.build("${DOCKER_IMAGE}:latest")
                }
            }
        }

        stage('Push to Docker Hub') {
            steps {
                withCredentials([usernamePassword(credentialsId: 'dockerhub-creds', usernameVariable: 'DOCKER_USER', passwordVariable: 'DOCKER_PASS')]) {
                    script {
                        sh 'echo "$DOCKER_PASS" | docker login -u "$DOCKER_USER" --password-stdin'
                        dockerImage.push()
                        dockerImage.push('latest')
                    }
                }
            }
        }

        stage('Deploy to EC2') {
            steps {
                sshagent(['ec2-ssh-key']) {
                    sh """
                        ssh -o StrictHostKeyChecking=no ubuntu@108.130.99.37 << 'ENDSSH'
                        docker pull ${DOCKER_IMAGE}:latest
                        docker stop resume-classifier-api || true
                        docker rm resume-classifier-api || true
                        docker run -d --name resume-classifier-api -p 80:8000 ${DOCKER_IMAGE}:latest
                        ENDSSH
                    """
                }
            }
        }
    }
}
