(use-modules (gnu packages base)
             (gnu packages rsync)
             (gnu packages ssh))

(packages->manifest
 (list coreutils
       openssh
       rsync
       sshpass))
