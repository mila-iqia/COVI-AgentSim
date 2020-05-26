conda activate {env_name}

cd {code_loc}

echo $(pwd)
echo $(which python)

echo "Stating Job"

date=$(date '+%Y-%m-%d %H:%M:%S')
name="_server_out.txt"

python server_bootstrap.py -e {weights} -w 4 > "$HOME/$date$name" 2>&1 &

# DO NOT WRITE ANYTHIN AFTER THIS, PYTHON WILL APPEND TO THIS FILE