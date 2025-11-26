import Link from "next/link"

export default function Footer() {
  return (
    <footer className="border-t bg-background">
      <div className="container py-8 md:py-12">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          <div>
            <Link href="/" className="flex items-center gap-2">
              <span className="bg-gradient-to-r from-violet-600 via-blue-500 to-teal-400 bg-clip-text text-xl font-bold text-transparent">
                EasyFlow ML
              </span>
            </Link>
            <p className="mt-2 text-sm text-muted-foreground">
              Automated machine learning platform for everyone. Build, train, and evaluate ML models without code.
            </p>
          </div>
          <div>
            <h3 className="text-lg font-semibold mb-4">Quick Links</h3>
            <ul className="space-y-2">
              <li>
                <Link href="/" className="text-sm text-muted-foreground hover:text-primary">
                  Home
                </Link>
              </li>
              <li>
                <Link href="/build" className="text-sm text-muted-foreground hover:text-primary">
                  Build Model
                </Link>
              </li>
              <li>
                <Link href="/about" className="text-sm text-muted-foreground hover:text-primary">
                  About
                </Link>
              </li>
            </ul>
          </div>
          <div>
            <h3 className="text-lg font-semibold mb-4">Contact</h3>
            <p className="text-sm text-muted-foreground">
              Have questions or feedback? Reach out to us at{" "}
              <a href="mailto:info@easyflowml.com" className="text-primary hover:underline">
                info@easyflowml.com
              </a>
            </p>
          </div>
        </div>
        <div className="mt-8 border-t pt-8 text-center text-sm text-muted-foreground">
          <p>© {new Date().getFullYear()} EasyFlow ML. All rights reserved.</p>
        </div>
      </div>
    </footer>
  )
}
