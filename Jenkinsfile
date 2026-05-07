pipeline {
    agent any

    environment {
        SONAR_PROJECT_KEY = 'heart_disease_mlops'  
        SONAR_SCANNER_HOME = tool 'Sonarqube'
        DOCKER_HUB_REPO = "htai99/mlops-project"
        DOCKER_HUB_CREDENTIALS_ID = "mlops-dockerhub-token"
    }

    stages {

        stage('Checkout Github') {
            steps {
                echo 'Checking out code from GitHub...'
		        checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'github-token', url: 'https://github.com/htasoftware99/heart_disease_mlops.git']])
		    }
        }

        stage('SonarQube Analysis'){
			steps {
				withCredentials([string(credentialsId: 'sonarqube-token', variable: 'SONAR_TOKEN')]) {
    					
					withSonarQubeEnv('Sonarqube') {
    						sh """
						${SONAR_SCANNER_HOME}/bin/sonar-scanner \
						-Dsonar.projectKey=${SONAR_PROJECT_KEY} \
						-Dsonar.sources=. \
						-Dsonar.host.url=http://sonarqube-dind:9000 \
						-Dsonar.login=${SONAR_TOKEN}
						"""
					}
				}
			}
		}

        stage('Build Docker Image') {
            steps {
                script {
                    echo 'Building Docker image...'
                    dockerImage = docker.build("${DOCKER_HUB_REPO}:latest")
                }
            }
        }

        stage('Trivy Security Scan') {
            steps {
                script {
                    echo 'Scanning the newly built image with Trivy...'
                    sh """
                    docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
                    aquasec/trivy:latest image --severity HIGH,CRITICAL ${DOCKER_HUB_REPO}:latest || true
                    """
                }
            }
        }

        stage('Push Image to DockerHub') {
            steps {
                script {
                    echo 'Pushing Docker image to DockerHub...'
                    docker.withRegistry('https://registry.hub.docker.com' , "${DOCKER_HUB_CREDENTIALS_ID}") {
                        dockerImage.push('latest')
                    }
                }
            }
        }

        stage('Install Kubectl & ArgoCD CLI') {
            steps {
                echo 'Installing Kubectl and ArgoCD CLI...'
            }
        }

        stage('Apply Kubernetes & Sync App with ArgoCD') {
            steps {
                echo 'Applying Kubernetes and syncing with ArgoCD...'
            }
        }
    }

}

