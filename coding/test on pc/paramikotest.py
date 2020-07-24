import paramiko
# import subprocess
# import os

# subprocess.call("ssh robot@192.168.137.3 python3 test/test2.py")

# process=subprocess.Popen(["cmd.exe","ssh robot@192.168.137.3 python3 test/test2.py"],stdout=subprocess.PIPE);
# result=process.communicate()[0]
# print (result)

ssh = paramiko.SSHClient()
ssh.connect('192.168.137.3', username='robot', password='maker')
ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command("'python3 test on pc/test2.py'")