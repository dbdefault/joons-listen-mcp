# Security Policy

## Supported versions

| Version | Supported |
| --- | --- |
| 0.1.x | Yes |
| older | No |

## Reporting a vulnerability

Report privately via **GitHub Security Advisories**: on this repository's **Security** tab, choose **Report a vulnerability**. Do not open a public issue or pull request for a security problem.

Please include the affected version, a minimal reproduction, and your assessment of the impact.

## Security model

- **Loopback only.** The HTTP service refuses to bind to anything but a loopback address and rejects requests whose `Host` header is not a loopback name or that carry an `Origin` header (blocks other web pages and DNS rebinding). The MCP server only calls a loopback `JOONS_LISTEN_URL` and ignores proxy environment variables.
- **No authentication.** Anyone who can open a connection to the loopback port, which means any program running as any user on the same machine, can submit jobs and read results. Do not expose the port with a tunnel or reverse proxy.
- **Allowed folders.** The service reads audio only from allowed folders (`JOONS_LISTEN_ROOTS`). Paths are resolved before the check, relative paths and files with secret-looking names are refused. Do not allow a folder that holds secrets.
- **Meeting content stays local.** Transcripts are sent only to a local LLM at a loopback URL (`JOONS_LISTEN_LLM_URL`). A non-loopback URL is rejected. The service never calls a cloud API with your audio or text. Model weights are downloaded from Hugging Face on first use; that is the only network traffic.
- **Data at rest is not encrypted.** Audio copies, transcripts, minutes and the dictionary live unencrypted in the data folder (`JOONS_LISTEN_DATA`). Protect that folder with your operating system's permissions. Use `JOONS_LISTEN_MIRROR_DIR` only for folders you trust.
- **Dictionary and hints.** The private dictionary is used as recognition hints and replacement rules. It never leaves the machine, and a filter removes hint lists the recognizer echoes back as text.
- **Extracted facts are never auto-confirmed.** Prices, quantities, dates and codes carry a `verify` flag.
- **Resource limits.** Request bodies, result chunk sizes and job lists are capped.
