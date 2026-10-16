// CI for bioclip-colombia, on a Jenkins running on the machine with the GPU.
// The test photos aren't in git, so BIOCLIP_HOME points to a checkout that has them (and the shared mlflow.db).
pipeline {
    agent any

    parameters {
        choice(name: 'MODEL', choices: ['bioclip', 'clip'], description: 'clip is the worse model: pick it to watch the gate block it')
        booleanParam(name: 'BUILD_IMAGE', defaultValue: false, description: 'also build the API Docker image')
    }

    environment {
        PATH = "${env.HOME}/.local/bin:${env.PATH}"
        // parameters only exist after the first run reads this file
        MODEL = "${params.MODEL ?: 'bioclip'}"
    }

    stages {
        stage('Install') {
            steps {
                sh 'uv sync --group mlops'
            }
        }

        stage('Unit tests') {
            steps {
                sh 'mkdir -p reports && uv run pytest -q --junitxml=reports/pytest.xml'
            }
            post {
                always { junit 'reports/pytest.xml' }
            }
        }

        stage('Web build') {
            steps {
                dir('web') {
                    sh 'npm ci && npm run lint && npm run build'
                }
            }
        }

        stage('Model quality gate (GPU)') {
            steps {
                sh '''
                    uv run --group mlops scripts/quality_gate.py \
                        --model "$MODEL" \
                        --testset "${BIOCLIP_HOME:-.}/data/testset" \
                        --tracking-uri "sqlite:///${BIOCLIP_HOME:-$PWD}/mlflow.db" \
                        --register
                '''
            }
        }

        stage('Docker image') {
            when { expression { params.BUILD_IMAGE } }
            steps {
                sh 'docker build -t bioclip-colombia-api:$BUILD_NUMBER .'
            }
        }
    }
}
