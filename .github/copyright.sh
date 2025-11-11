#!/bin/bash

# If there are new files with headers that can't match the conditions here,
# then the files can be ignored by an additional glob argument via the -g flag.
# For example:
#   -g "!src/special_file.rs"
#   -g "!src/special_directory"

# Check all the standard Rust source files (non-Solvi)
output=$(rg "^// Copyright (19|20)[\d]{2} (.+ and )?the Fiksi Authors( and .+)?$\n^// SPDX-License-Identifier: Apache-2\.0 OR MIT$\n\n" --files-without-match --multiline -g "*.rs" -g "!solvi/*" -g "!colamd_rs/*" .)

if [ -n "$output" ]; then
	echo -e "The following files lack the correct copyright header:\n"
	echo $output
	echo -e "\n\nPlease add the following header:\n"
	echo "// Copyright $(date +%Y) the Fiksi Authors"
	echo "// SPDX-License-Identifier: Apache-2.0 OR MIT"
	echo -e "\n... rest of the file ...\n"
	exit 1
fi

# Check all the standard Rust source files (for Solvi)
output=$(rg "^// Copyright (19|20)[\d]{2} (.+ and )?the Solvi Authors( and .+)?$\n^// SPDX-License-Identifier: Apache-2\.0 OR MIT$\n\n" --files-without-match --multiline -g "solvi/**/*.rs" .)

if [ -n "$output" ]; then
	echo -e "The following files lack the correct copyright header:\n"
	echo $output
	echo -e "\n\nPlease add the following header:\n"
	echo "// Copyright $(date +%Y) the Solvi Authors"
	echo "// SPDX-License-Identifier: Apache-2.0 OR MIT"
	echo -e "\n... rest of the file ...\n"
	exit 1
fi

# Check all the standard Rust source files (for colamd_rs)
output=$(rg "^// COLAMD, Copyright \\(c\\) 1998-2024, Timothy A. Davis and Stefan Larimore,\n^// All Rights Reserved.\n^// Copyright (19|20)[\d]{2} (.+ and )?the Solvi Authors( and .+)?$\n^// SPDX-License-Identifier: BSD-3-Clause$\n\n" --files-without-match --multiline -g "colamd_rs/**/*.rs" .)

if [ -n "$output" ]; then
	echo -e "The following files lack the correct copyright header:\n"
	echo $output
	echo -e "\n\nPlease add the following header:\n"
	echo "// COLAMD, Copyright (c) 1998-2024, Timothy A. Davis and Stefan Larimore,"
	echo "// All Rights Reserved."
	echo "// Copyright $(date +%Y) the Solvi Authors"
	echo "// SPDX-License-Identifier: BSD-3-Clause"
	echo -e "\n... rest of the file ...\n"
	exit 1
fi

echo "All files have correct copyright headers."
exit 0

