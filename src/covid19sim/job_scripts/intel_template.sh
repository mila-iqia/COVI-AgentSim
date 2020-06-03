conda activate {env_name}

cd {code_loc}

echo $(pwd)
echo $(which python)

echo "Stating Job"

use_transformer={use_transformer}

if [ "$use_transformer" = true ] ; then
    python server_bootstrap.py -e {weights} -w {workers} {frontend} {backend}&
fi

# DO NOT WRITE ANYTHIN AFTER THIS, PYTHON WILL APPEND TO THIS FILE