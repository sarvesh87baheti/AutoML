import type { LucideIcon } from "lucide-react"

interface FeatureCardProps {
  icon: LucideIcon
  title: string
  description: string
  iconColor: string
}

export default function FeatureCard({ icon: Icon, title, description, iconColor }: FeatureCardProps) {
  return (
    <div className="flex flex-col items-center text-center p-6 rounded-lg border bg-card">
      <div className="rounded-full bg-gradient-to-br from-violet-600/20 via-blue-500/20 to-teal-400/20 p-3 mb-4">
        <Icon className={`h-6 w-6 text-${iconColor}`} />
      </div>
      <h3 className="text-xl font-bold mb-2">{title}</h3>
      <p className="text-muted-foreground">{description}</p>
    </div>
  )
}
