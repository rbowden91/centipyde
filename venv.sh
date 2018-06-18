if [ ! -d venv ]; then
    virtualenv --python=python3.6 venv
fi
. ./venv/bin/activate
