# This script processes the Markdown file downloaded from a Jupyter notebook
# and prepares it for deployment through the static site generator Jekyll.

r_file_name = '_posts/2018-12-18-motion_by_mean_curvature.md'
in_fid = open(r_file_name)
w_file_name = '_posts/2018-12-18-motion_by_mean_curvature_rev.md'
out_fid = open(w_file_name, 'w')
str = in_fid.read()
str.replace('\begin{equation*}', '\\[')
str.replace('\end{equation*}', '\\]')
str = str.replace('\\\\', '\\\\\\')
str = str.replace('\\', '\\\\')
str = str.replace('./media/motion_by_mean_curvature/images/', '/media/posts/motion_by_mean_curvature/')
out_fid.write(str)
out_fid.close()
in_fid.close()
