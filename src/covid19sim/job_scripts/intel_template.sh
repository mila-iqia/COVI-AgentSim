conda activate {env_name}

cd {code_loc}

echo $(pwd)
echo $(which python)

echo "Stating Job"


python server_bootstrap.py -e {weights} -w 4 --frontend-port=6666 --backend-port=6667 > {server_out} 2>&1 &

# DO NOT WRITE ANYTHIN AFTER THIS, PYTHON WILL APPEND TO THIS FILE